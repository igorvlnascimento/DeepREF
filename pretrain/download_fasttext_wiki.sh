mkdir pretrain/fasttext_wiki
wget -P pretrain/fasttext_wiki https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip pretrain/fasttext_wiki/wiki-news-300d-1M.vec.zip -d pretrain/fasttext_wiki
rm pretrain/fasttext_wiki/wiki-news-300d-1M.vec.zip