# Contributing to PVTModel

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to PVTModel, the repository associated with the `heat-panel` Python package. These guidelines are hosted on GitHub and are maintained, so ensure you have the latest version via a `git pull`. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

#### Table Of Contents

[Code of Conduct](#code-of-conduct)

[What should I know before I get started?](#what-should-i-know-before-i-get-started)
  * [What is PVTModel for?](#what-is-pvtmodel-for)
  * [What is PVTModel not for?](#what-is-pvtmodel-not-for)

[How to contribute to PVTModel](#how-to-contribute-to-PVTModel)
  * [Reporting bugs](#reporting-bugs)
  * [Merging patches](#merging-patches)
    * [Cosmetic patches](#cosmetic-patches)
  * [Questions](#questions)

[Styleguides](#styleguides)
  * [Git commit messages](#git-commit-messages)
  * [Python styleguide](#python-styleguide)
  * [YAML styleguide](#yaml-styleguide)
  * [Changing styleguides](#changing-styleguides)


### Code of Conduct

This project and everyone participating in it is governed by the [PVTModel Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to @BenWinchester.

### What should I know before I get started?

PVTModel was developed at Imperial College London as a means of simulating the electrical and thermal characteristics of hybrid photovoltaic-thermal collectors. The name, `HEATPanel`, refers to this: Hybrid Electricity And Thermal Panel. Primarilly, sheet-and-tube collectors have been investigated, and default data for these panels is provided which can be adjusted by the user.

PVTModel has the capability to model these panels in a great level of detail, considering layer- and element-wise thermal and electrical characteristics. These are then aggregated to provide overall performance features which are useful when considering these types of collectors.

#### What is PVTModel for?

PVTModel is a software tool for simulating the electrical and thermal characteristics of hybrid PV-T collectors, specifically sheet-and-tube collectors, though support for more collectors is being developed as part of the ongoing process of expanding the functionality of the software package. These collectors can be considered as part of a wider hot-water demand system, including storage tanks and heat exchangers, or as standalone collectors under test conditions. 2D temperature maps of each layer can be generated for vieweing the collectors' thermal properties.

#### What is PVTModel not for?

Fundamentally, PVTModel is a matrix solver with bells and whistles added to ensure a user-friendly process for seamlessly adjusting parameters without having to recreate a matrix equation manually for each calculation that a user could wish to solve. With this in mind, chosing parameters that are outside a standard scope of a matrix solver may lead to divergent solutions.

PVTModel is also currently only developed for sheet-and-tube PV-T collectors. This is a great limitation in terms of the available market of PV-T collectors which currently exist, and the multitude of designs in development. However, sheet-and-tube collectors are currently the most widely implemented.

## How to contribute to PVTModel

### Reporting bugs

**Did you find a bug?** Bugs make it into the code from time to time. If you spot a bug, report it as follows:

* **Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/BenWinchester/PVTModel/issues).

* If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/BenWinchester/PVTModel/issues/new/choose). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

  * If the issue is a **bug**, use the [Bug report](BenWinchester/PVTModel/issues/new?assignees=&labels=bug&template=bug_report.md&title=) template,

  * If the issue is a **feature** request for something new that you would like to see introduced into PVTModel, use the [Feature request](BenWinchester/PVTModel/issues/new?assignees=&labels=bug&template=feature_request.md&title=) template.

### Merging patches

**Did you write a patch that fixes a bug?** If you have coded a solution for a bug that you have found or for an open issue, open a pull request for it as follows:

* Open a new GitHub pull request with the patch.

* Ensure the PR description clearly describes the problem and solution:

  * Include the relevant issue number if applicable,

  * Follow the template information presented, filling in all the fields requested which are relevant to your patch.

* Ensure that you include at least one administrator reviewer for your pull request. Without an appropriate review, you will be unable to merge your pull request.

#### Cosmetic patches

**Did you fix whitespace, format code, or make a purely cosmetic patch?** Changes that are cosmetic in nature and do not add anything substantial to the stability, functionality, or testability of PVTModel will generally not be accepted. Contact the developers directly, or save your cosmetic changes until you are able to merge them as part of a feature or bug issue.

### Questions

**Do you have questions about the source code?** Ask any question about how to use PVTModel on the [Discussions](https://github.com/BenWinchester/PVTModel/discussions) page.

## Styleguides

### Git commit messages

* Git Commit Messages
* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* When only changing documentation, include [ci skip] in the commit title
* Consider starting the commit message with an applicable emoji:
  * 🎉 `:tada` when adding an initial commit or solving a difficult problem
  * 🎨 `:art:` when improving the format/structure of the code
  * 🐎 `:racehorse:` when improving performance
  * 📝 `:memo:` when writing docs
  * 🐧 `:penguin:` when fixing something on Linux
  * ✨ `:sparkles:` when adding a new feature
  * 🚧 `:construction:` when part-way through working on code
  * 🍎 `:apple:` when fixing something on macOS
  * 🏁 `:checkered_flag:` when fixing something on Windows
  * ⏪ `:rewind:` when backing out a commit or changes
  * 🐛 `:bug:` when fixing a bug
  * 🔥 `:fire:` when removing code or files
  * 💚 `:green_heart:` when fixing the CI build
  * 👕 `:shirt:` when removing linter warnings
  * ✅ `:white_check_mark:` when adding tests
  * 🔒 `:lock:` when dealing with security
  * ⬆️ `:arrow_up:` when upgrading dependencies
  * ⬇️ `:arrow_down:` when downgrading dependencies
  * 🚀 `:rocket:` when deploying code

### Python styleguide

All `Python` code must conform to [mypy](https://github.com/python/mypy) and [pylint](https://github.com/PyCQA/pylint) coding standards and must be formatted with the [black](https://github.com/psf/black) formatter:
* A `mypy.ini` file within the root of the repository sets out the requirements that must be met by any code. Use `python -m mypy src/` to ensure that your code complies with the guidelines.
* A `.pylintrc` file within the root of the repository sets out the linting requirements. Use `python -m pylint src/` to ensure that your code complies with the guidelines.
* All code must be formatted with `black`.

These tests must pass for any pull request to be successfully approved and merged. You can run these tests from the root of the repository with `./bin/test-clover.sh`.

### YAML styleguide

All `.yaml` files which are modified are linted with [yamllint](https://github.com/adrienverge/yamllint). You can use `yamllint -c .yamllint-config.yaml` to run `yamllint` over any `.yaml` files that you have modified.

### Changing styleguides

If you have any changes for the styleguides, make these **very clear** within your pull request message. It is unusual to have to change the styleguides to make your code pass the required tests.

Thanks! :heart: :heart: :heart:

PVTModel Team
