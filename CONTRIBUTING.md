# Contributing to Open AVM Kit

Thank you for your interest in contributing to **Open AVM Kit**! Contributions are what make open-source projects thrive, and we value your involvement.

## Contributor License Agreement (CLA)

Before we can accept your contributions, you must agree to our Contributor License Agreement (CLA). This ensures we can re-license contributions as part of the dual-license model. You can find the contributor license agreement here:
https://docuseal.com/d/4iF7pn89f5v914

See [License Philosophy](LICENSE-PHILOSOPHY.md) for more information on why we use a dual-license model.

### How to Sign the CLA
- **Once, before your first contribution:** sign the CLA on [DocuSeal](https://docuseal.com/d/4iF7pn89f5v914), entering your GitHub username in the form so a maintainer can match your signature to your GitHub account.
- **On every pull request:** tick the box *"I have read and agree to the Contributor License Agreement"* in the pull request description (it's in the template).
- A maintainer will verify your DocuSeal signature and record it. After that, your future pull requests only need the checkbox.
- Make sure your commits are authored with an email address linked to your GitHub account, so your commits can be matched to your signature.

The **CLA** check on your pull request will tell you what, if anything, is still missing.

## Contribution Guidelines

1. **Fork the Repository**
   - Create your own fork of the project on GitHub.

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/your-username/your-project-name.git
   cd your-project-name
   ```

3. **Create a Branch**
   - Use a descriptive branch name for your changes:
   ```bash
   git checkout -b feature/my-new-feature
   ```

4. **Make Changes**
   - Follow the coding style and conventions of the project.
   - Write clear, concise commit messages.
   
5. **Run Tests**
   - Ensure all existing and new tests pass before submitting your changes. The test tools aren't installed with OpenAVMKit itself; install them once with the `dev` extra:
   ```bash
   pip install -e ".[dev]"
   pytest tests/
   ```

6. **Open a Pull Request**
   - Push your branch to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```
   - Open a pull request against the main repository. Include a clear title and description of your changes.

## Style Guide
   - Follow PEP 8 for Python Code
   - Write clear and concise comments where necessary
   - Document all public-facing functions and classes using docstrings

We look forward to your contributions! 🎉