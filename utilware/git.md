-  **local repo**
-  **remote repo**

`git help <command>`
`git <command> --help`

## working dir -> staging area -> local repo

`git init` -> in the project directory. makes a ".git" folder-> the **local repo**

`git add < >` (eg: *) -> add to stage
`git add -A` -> stage all changes

`git rm < >` -> removes from stage + deletes in working dir
`git rm --cached < >` -> ("unstage") removes from stage, won't delete  

`git status` -> files staged, unstaged  

`git commit -m "message"` -> saves all in stage as a 'version'  

`git log` -> log of commits  

`git diff HEAD` -> diff between working dir and last commit
`git diff --cached` -> diff between stage and last commit  

"HEAD" => active branch
"cache" => stage

### git ignore

- create a file with name ".gitignore" - list the files to be ignored while staging("adding" to stage). eg:
\# comment
\# ignore all such files(regex)
*.jpg
\# ignore all files in this dir
data/
- paths in gitignore are relative. Can have multiple gitignore files. Add gitignore file in a folder with "*" in it - to not add anything in that folder to stage.

### branching

**branch = chain of commits**

`git branch <branch_name>` -> create branch

`git checkout <branch_name>` -> move to that branch

`git merge <branch_name>` -> merge that branch to "current"(checked out) branch  

"master" = default main branch

(change main repo to "master" if not so - in account)  

## "Remote"

### user (github account)

`git config --global user.name "username"` -> user(account) to be used for all commits
`git config --global user.email "email@example.com"`

`git config --list`

'global' flag => config for the linux user [need only be done **once** in the system by that user - that config for all repos]   
without 'global' flag => only for that repo

### remote repo
`git remote add <remote-alias> <url>` -> connects remote and local
( remote will accessed with that alias henceforth (alias for the url) )
make a repository in github and use its url  

`git remote -v` -> show remote info  

`git pull <remote-alias> <branch>` -> pull from remote (that branch) - syncs local with remote (files, .git, etc. updated)
`git push <remote-alias> <branch>` -> push "current" branch to remote (given branch) - syncs remote with local
  
`git fetch <remote-alias> <branch>` -> downloads but won't merge with local-master (commits to a temporary branch instead)
pull = fetch + merge

`git clone <url>` -> downloads the whole repo (this has nothing to do with a local repo) - the downloaded thing will be used as the local side then(eg: the .git in the downloaded will be the local repo - it will already have info about the remote link - so don't have to connect remote and local (ie, "add remote") in this case)

`git clone = mkdir + git init + git remote add + git pull`

cloning => only the master branch

fork -> github feature to save a copy of a repo in our account
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI2MzY2MDMzXX0=
-->