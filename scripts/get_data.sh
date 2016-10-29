curl -o ../data/data.tar.gz http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
mkdir ../data
tar zvfx ../data/data.tar.gz -C ../data/
rm ../data/data.tar.gz
