// CLA check for pull requests. Run by .github/workflows/cla.yml via actions/github-script.
//
// A PR passes when:
//   1. the PR author ticked the CLA checkbox in the PR description (bots are exempt), and
//   2. every GitHub user who authored a commit in the PR is recorded in
//      signatures/version1/cla.json on the default branch.
//
// Contributors sign the CLA once on DocuSeal. A maintainer checks DocuSeal and then applies
// the `cla-verified` label; on that label event this script records the PR's unrecorded
// commit authors in cla.json, so their later PRs pass without a label.
//
// The signature file is always read from the default branch through the API, never from the
// PR, so a PR cannot add itself to the list.

const MARKER = '<!-- cla-check -->';
const LABEL = 'cla-verified';
const SIG_PATH = 'signatures/version1/cla.json';
const DOCUSEAL_URL = 'https://docuseal.com/d/4iF7pn89f5v914';
const CLA_URL = 'https://github.com/larsiusprime/openavmkit/blob/master/CLA.md';
const CHECKBOX_RE =
  /^\s*[-*]\s*\[[xX]\]\s*I have read and agree to the \[?Contributor License Agreement/m;

const isBot = (user) =>
  !!user && (user.type === 'Bot' || (user.login || '').endsWith('[bot]'));

function checkboxTicked(body) {
  return CHECKBOX_RE.test(body || '');
}

// Split the PR's commits into GitHub users and commits with no linked GitHub account.
function collectAuthors(commits, prAuthor) {
  const users = new Map();
  const unlinked = new Map();
  const add = (u) => {
    if (u && !isBot(u) && !users.has(u.id)) users.set(u.id, { login: u.login, id: u.id });
  };
  add(prAuthor);
  for (const c of commits) {
    if (c.author) {
      add(c.author);
    } else {
      const a = (c.commit && c.commit.author) || {};
      const key = (a.email || a.name || c.sha).toLowerCase();
      if (!unlinked.has(key)) unlinked.set(key, { name: a.name, email: a.email, sha: c.sha });
    }
  }
  return { users: [...users.values()], unlinked: [...unlinked.values()] };
}

function buildComment({ needsCheckbox, prAuthor, missing, unlinked }) {
  const lines = [MARKER, '### Contributor License Agreement', ''];
  if (!needsCheckbox && missing.length === 0 && unlinked.length === 0) {
    lines.push('✅ CLA requirements are met. Thank you!');
    return lines.join('\n');
  }
  lines.push(
    'Thank you for your contribution! Before it can be merged:',
    ''
  );
  if (needsCheckbox) {
    lines.push(
      `- [ ] @${prAuthor} — please edit the PR description and tick the box ` +
        `**"I have read and agree to the Contributor License Agreement"** ` +
        `([read the CLA](${CLA_URL})). If the box is missing, add this line:\n` +
        '  `- [x] I have read and agree to the Contributor License Agreement for this contribution.`'
    );
  }
  if (missing.length > 0) {
    lines.push(
      `- [ ] These contributors need to sign the CLA once, here: **[sign on DocuSeal](${DOCUSEAL_URL})** ` +
        '(enter your GitHub username in the form). A maintainer will then verify it — no further action needed in this thread.',
      ...missing.map((u) => `  - @${u.login}`)
    );
  }
  if (unlinked.length > 0) {
    lines.push(
      '- [ ] These commits are authored with an email that is not linked to a GitHub account, ' +
        'so the signature cannot be matched. Add the email to your GitHub account ' +
        '(Settings → Emails) or re-author the commits, then push again:',
      ...unlinked.map((u) => `  - ${u.name || 'unknown'} <${u.email || '?'}> (${u.sha.slice(0, 7)})`)
    );
  }
  lines.push(
    '',
    '<sub>Maintainers: after confirming the DocuSeal signature(s) for everyone listed above, ' +
      `apply the \`${LABEL}\` label to record them.</sub>`
  );
  return lines.join('\n');
}

async function readSignatures(github, owner, repo, ref) {
  const { data } = await github.rest.repos.getContent({ owner, repo, path: SIG_PATH, ref });
  const json = JSON.parse(Buffer.from(data.content, 'base64').toString('utf8'));
  json.signedContributors = json.signedContributors || [];
  return { json, sha: data.sha };
}

async function recordSignatures({ github, owner, repo, ref, users, pr, verifiedBy, core }) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const { json, sha } = await readSignatures(github, owner, repo, ref);
    const known = new Set(json.signedContributors.map((s) => s.id));
    const toAdd = users.filter((u) => !known.has(u.id));
    if (toAdd.length === 0) return json;
    const now = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
    for (const u of toAdd) {
      json.signedContributors.push({
        name: u.login,
        id: u.id,
        created_at: now,
        repoId: pr.base.repo.id,
        pullRequestNo: pr.number,
        method: 'docuseal',
        verified_by: verifiedBy,
      });
    }
    try {
      await github.rest.repos.createOrUpdateFileContents({
        owner,
        repo,
        path: SIG_PATH,
        branch: ref,
        sha,
        message: `Record CLA signature(s) for PR #${pr.number}: ${toAdd.map((u) => u.login).join(', ')}`,
        content: Buffer.from(JSON.stringify(json, null, 2) + '\n').toString('base64'),
      });
      core.info(`Recorded: ${toAdd.map((u) => u.login).join(', ')}`);
      return json;
    } catch (e) {
      if (e.status !== 409 || attempt === 2) throw e;
      core.info('cla.json changed underneath us; retrying.');
    }
  }
}

async function upsertComment({ github, owner, repo, number, body, failing }) {
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number: number,
    per_page: 100,
  });
  const existing = comments.find((c) => c.body && c.body.startsWith(MARKER));
  if (existing) {
    if (existing.body !== body) {
      await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body });
    }
  } else if (failing) {
    // Only start a thread when something is missing; passing PRs stay quiet.
    await github.rest.issues.createComment({ owner, repo, issue_number: number, body });
  }
}

async function run({ github, context, core }) {
  const pr = context.payload.pull_request;
  const { owner, repo } = context.repo;
  const ref = context.payload.repository.default_branch;

  const commits = await github.paginate(github.rest.pulls.listCommits, {
    owner,
    repo,
    pull_number: pr.number,
    per_page: 100,
  });
  const { users, unlinked } = collectAuthors(commits, pr.user);

  let { json } = await readSignatures(github, owner, repo, ref);

  if (context.payload.action === 'labeled' && context.payload.label.name === LABEL) {
    const sender = context.payload.sender.login;
    const { data: perm } = await github.rest.repos.getCollaboratorPermissionLevel({
      owner,
      repo,
      username: sender,
    });
    if (['admin', 'maintain', 'write'].includes(perm.permission)) {
      json = await recordSignatures({ github, owner, repo, ref, users, pr, verifiedBy: sender, core });
    } else {
      core.warning(`${sender} applied ${LABEL} without write access; ignoring.`);
    }
  }

  const signed = new Set(json.signedContributors.map((s) => s.id));
  const missing = users.filter((u) => !signed.has(u.id));
  const needsCheckbox = !isBot(pr.user) && !checkboxTicked(pr.body);
  const failing = needsCheckbox || missing.length > 0 || unlinked.length > 0;

  const body = buildComment({ needsCheckbox, prAuthor: pr.user.login, missing, unlinked });
  await upsertComment({ github, owner, repo, number: pr.number, body, failing });

  if (failing) {
    const reasons = [];
    if (needsCheckbox) reasons.push('CLA checkbox not ticked');
    if (missing.length) reasons.push(`unrecorded signers: ${missing.map((u) => u.login).join(', ')}`);
    if (unlinked.length) reasons.push(`${unlinked.length} commit author(s) not linked to GitHub`);
    core.setFailed(`CLA check failed — ${reasons.join('; ')}`);
  } else {
    core.info('CLA requirements met.');
  }
}

module.exports = run;
module.exports._internal = { checkboxTicked, collectAuthors, buildComment, isBot };
