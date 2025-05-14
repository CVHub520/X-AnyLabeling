# X-AnyLabeling - Contributing Guide 🌟

We're thrilled that you want to contribute to X-AnyLabeling, the future of communication! 😄

X-AnyLabeling is an open-source project, and we welcome your collaboration. Before you jump in, let's make sure you're all set to contribute effectively and have loads of fun along the way!

## Table of Contents

- [Fork the Repository](#fork-the-repository)
- [Clone Your Fork](#clone-your-fork)
- [Create a New Branch](#create-a-new-branch)
- [Code Like a Wizard](#code-like-a-wizard)
- [Committing Your Work](#committing-your-work)
- [Sync with Upstream](#sync-with-upstream)
- [Open a Pull Request](#open-a-pull-request)
- [Review and Collaboration](#review-and-collaboration)
- [Celebrate 🎉](#celebrate-)

## Fork the Repository

🍴 Fork this repository to your GitHub account by clicking the "Fork" button at the top right. This creates a personal copy of the project you can work on.

## Clone Your Fork

📦 Clone your forked repository to your local machine using the `git clone` command:

```bash
git clone https://github.com/YourUsername/X-AnyLabeling.git
```

## Create a New Branch

🌿 Create a new branch for your contribution. This helps keep your work organized and separate from the main codebase.

```bash
git checkout -b your-branch-name
```

Choose a meaningful branch name related to your work. It makes collaboration easier!

## Google-Style Docstrings

For clarity and maintainability, any new functions or classes must include [Google-style docstrings](https://google.github.io/styleguide/pyguide.html) and use Python type hints. Type hints are mandatory in all function definitions, ensuring explicit parameter and return type declarations. These docstrings should clearly explain parameters, return types, and provide usage examples when applicable.

For example:

```python
def greet(name: str, greeting: str = "Hello") -> str:
    """
    Greets a person with a specified greeting.

    Args:
        name (str): The name of the person to greet.
        greeting (str): The greeting message to use, defaults to "Hello".

    Returns:
        str: A greeting message.

    Examples:
        >>> greet("World")
        'Hello, World!'
        >>> greet("CVHub", "Hi")
        'Hi, CVHub!'
    """
    return f"{greeting}, {name}!"
```

Following this pattern helps ensure consistency throughout the codebase.

## Code Like a Wizard

🧙‍♀️ Time to work your magic! Write your code, fix bugs, or add new features. Be sure to follow our project's coding style. You can check if your code adheres to our style using:

```bash
bash scripts/format_code.sh
```

This adds a bit of enchantment to your coding experience! ✨

## Committing Your Work

📝 Ready to save your progress? Commit your changes to your branch.

```bash
git add .
git commit -m "Your meaningful commit message"
```

Please keep your commits focused and clear. And remember to be kind to your fellow contributors; keep your commits concise.

## Sync with Upstream

⚙️ Periodically, sync your forked repository with the original (upstream) repository to stay up-to-date with the latest changes.

```bash
git remote add upstream https://github.com/CVHub520/X-AnyLabeling.git
git fetch upstream
git merge upstream/main
```

This ensures you're working on the most current version of X-AnyLabeling. Stay fresh! 💨

## Open a Pull Request

🚀 Time to share your contribution! Head over to the original X-AnyLabeling repository and open a Pull Request (PR). Our maintainers will review your work.

## Review and Collaboration

👓 Your PR will undergo thorough review and testing. The maintainers will provide feedback, and you can collaborate to make your contribution even better. We value teamwork!

## Celebrate 🎉

🎈 Congratulations! Your contribution is now part of X-AnyLabeling. 🥳

Thank you for making X-AnyLabeling even more magical. We can't wait to see what you create! 🌠

Happy Coding! 🚀🦄