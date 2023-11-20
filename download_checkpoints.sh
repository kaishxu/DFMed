# Download meddg generation checkpoint from Google Drive
filename='meddg_fine_tuned_checkpoint.zip'
fileid='1cIzbRu4Hb6IxMJZ_Ig8I25gtOSRE5aTO'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Unzip
unzip -q ${filename}
rm ${filename}

# Download kamed generation checkpoint from Google Drive
filename='kamed_fine_tuned_checkpoint.zip'
fileid='15ZwWVqkaugA7EFZEQDS-2B4f3LKgtpAA'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Unzip
unzip -q ${filename}
rm ${filename}

rm ./cookie
