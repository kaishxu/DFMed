# Download dual flow learning results from Google Drive
filename='df_results.zip'
fileid='1ZOaOjDYqqCV4bUyJBG6KA4eAh3zlqz6j'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Unzip
unzip -q ${filename}
rm ${filename}

# Download response generation results from Google Drive
filename='generation_results.zip'
fileid='1bCgziRag6uL_kDRzN49vJeJQ76iSQyo1'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Unzip
unzip -q ${filename}
rm ${filename}

rm ./cookie
