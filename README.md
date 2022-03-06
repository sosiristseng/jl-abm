# Jupyter book template for Julia Jupyter notebooks

## Features

- [Jupyter book](https://jupyterbook.org/index.html) builds `md` and `ipynb` files into a website.
- GitHub actions and GitLab CI/CD build and publish the website whenever changes are committed.
  - The notebook execution results are cached so you can push notebooks with output cell cleared and enjoy the results once the build action is completed.
- Periodically updating Julia dependencies and make a PR if notebooks are executed successfully.
  - For GitHub: you need a pair of SSH keys. (Public key: Deploy key; private key : `SSH_PRIVATE_KEY` actions secret)
  - For GitLab: you need a `GIT_PUSH_TOKEN` [CI/CD variable](https://docs.gitlab.com/ee/ci/variables/index.html), which is a PAT with `write_repo` access.

## Commands

### Install Julia dependencies without updating

```bash
julia --project=. --color=yes --threads=auto -e 'using Pkg; Pkg.instantiate()'
```

### Update Julia dependencies

```bash
julia --project=. --color=yes --threads=auto -e 'using Pkg; Pkg.update()'
```
