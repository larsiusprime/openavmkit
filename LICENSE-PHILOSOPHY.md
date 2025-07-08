# Licensing Philosophy for openavmkit

by Lars A. Doucet, [BDFL](https://en.wikipedia.org/wiki/Benevolent_dictator_for_life)

Dual licensing is controversial for some in the open source community, so I feel it is best to be completely transparent about my reasoning for choosing this model for openavmkit.

These are my goals:

- Be useful to governments, policymakers, researchers, students, and analysts
- Require commercial companies to contribute back to its development if they want to use it

Commercial companies are free to use the software for any purpose, they just need to follow the terms of the license. That generally means contributing back any code changes they make. Those who don't want to contribute back code changes can pay the maintainers some money or work out some other agreement. This is the dual licensing model I have chosen.

This isn't the only choice I could have made. Let's look at the alternatives:

- Permissive license (MIT/BSD)
- Copyleft license (GPL/AGPL)
- Commercial license
- Dual license (copyleft + commercial)

**Permissive licenses** are the simplest and easiest and I have used them in my prior open source work. These licenses explicitly allow anyone to do anything with the code -- including allowing commercial companies to take the code, add a bunch of proprietary stuff on top, and contribute nothing back. I am not philosophically opposed to this in general--I have worked on many permissively licensed libraries that commercial companies made use of with my full blessing--but is counter to my goals for this specific project.

**Copyleft licenses** are very close to what I want, but I hesitate because there's no going back on a pure copyleft license. By contrast, you are always free to change your mind later with a permissive license or a commercial license. Copyleft licenses also tend to scare commercial companies away completely. And although I want to gently discourage commercial free-riders, I don't want to shut out the commercial sector completely.

There's another subtle consideration, given my target user base. A lot of governments have private contracts with commercial vendors and might want to build integrations for openavmkit into the proprietary software they use every day. This would require the legal wiggle room to create an exception. Since my primary goal is to be useful to these kinds of users, I am hesitant to use a pure copyleft license that will permanently wall off this sort of use case.

**Commercial licenses** would mean keeping the entire codebase proprietary, and I don't want to do that. I am not running openavmkit as a business, even if I am happy to allow the occasional commercial licensee to compensate us monetarily in lieu of contributing back code changes.

# Balancing pros and cons

Here you can see a table balancing the pros and cons:

| License      | Public<br>appeal | Commmercial<br>appeal | Enforce<br>contribution | Low<br>friction | Proprietary<br>integration | Can<br>relicense |
|--------------|------------------|-----------------------|-------------------------|-----------------|----------------------------|------------------|
| Permissive   | ✅              | ✅                     | ❌                    | ✅              | ✅                        | ✅                |
| Copyleft     | ✅              | ❌                     | ✅                    | ✅              | ❌                        | ❌                |
| Commercial   | ❌              | ✅                     | ❌                    | ❌              | ✅                        | ✅                |
| Dual-license | ✅              | ✅                     | ✅                    | ❌              | ✅                        | ✅                |

These are the reasons for and against each license, as I happen to see them. You might feel differently (for instance, for some people "can relicense" is a con rather than a pro). As you can see, there is no single license that ticks every single box. 

Given my goals, the **dual-licensed** model of shipping simultaneously under an aggressive copyleft license alongside an optional commercial license best fulfills my objectives.

That doesn't mean there aren't downsides, even from my own perspective. Most notably, there are many developers who categorically refuse to sign the contributor license agreement (CLA) mandated by the dual-licensed model.

These are all valid concerns, and I respect those who feel that way. To these I must add this: opinions aside, the CLA adds extra friction to the contribution process, and this will discourage some contributors all by itself.

I am willing to accept this downside for a few reasons:

- I am being clear about my goals and intentions from the very beginning
- I expect I will be contributing most of the code to this project
- In the event this was a huge mistake, the dual-license model gives me the freedom to pick a different license that better reflects what's best for its users

# Who is "I?"

You'll notice I'm writing in the first person here. That is because I, Lars A. Doucet, author and maintainer of this library, am the principal copyright holder and will continue to be so throughout its life. I am not a corporation, I am not a committee, I am not a foundation. I am a single individual who is making decisions about this project based on my own personal values and goals. It is true that I work with many other people and organizations (most notably the Center for Land Economics, who is generously sponsoring my work) but as far as the openavmkit library in particular is concerned, the buck stops with me. Other libraries I work on will be governed by their own licenses and policies. 

# A note on your rights

You have rights under the published license that no one can ever take away. 

You can always use the latest version of the software that is licensed with the AGPLv3 under those terms. That means, among other things, that you never have to pay to use the software.  

Also, you will always have the right to fork the AGPLv3 version of the project and continue development under the AGPLv3. This holds even if I one day decide to re-license the entire thing under some other license. You will always be able to use the software for any purpose, including commercial purposes, under the terms of the AGPLv3.

You also have the right to not sign any contributor license agreement. The only thing not signing would prevent you from doing is having your changes merged into this upstream repository. Other forks of this repository are free to accept changes under the AGPLv3 without any further agreement.

It's true that many companies are scared away by aggressive copyleft terms (and I am intentionally incorporating a copyleft license for that very reason) but let's recognize that such reticence is a *business decision* and that plenty of companies are happy to use and contribute back to AGPLv3-licensed projects.
 
