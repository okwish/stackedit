# Setup

  

## github account config (once in system)

`git config --global user.name "username"` -> user(account) to be used for all commits
`git config --global user.email "email@example.com"`


## authentication (once - till expiry)  

- settings - developer setting(bottom left) - personal access tokens - classic - generate new token(general use) - save token somewhere

- the token is to be used instead of password

## misc

(change main repo to "master" if not so in account)

settings - repositories(left) - default branch

# project

## (once for a repo):

- mkdir
-  `git init`
- make a repo in github account
-  `git remote add <alias-for-remote> <url>`

(use repo-name_remote as alias)

## (then)

- do stuff, `git add -A`, `git commit -m "message"`
-  `git pull <remote-alias> <branch>`, `git push <remote-alias> <branch>`

(.gitignore if needed)  

## new place

`git clone = mkdir + git init + git remote add + git pull`
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTY4NDkyMzU5MF19
-->