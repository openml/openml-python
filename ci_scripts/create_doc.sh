set -o pipefail

# delete any previous documentation folder for the branch
# if it exists
if [ -d doc/$1 ]; then
    rm -rf doc/$1
fi


# create the documentation
cd doc && make html 2>&1 | tee ~/log.txt

# create directory with branch name
mkdir $1

# copy content
cp - r build/html/* $1
