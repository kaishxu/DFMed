# Download dataset from Google Drive
filename='data.zip'
fileid='1Tb2tO3tQ-6a-j0AiBKDu0RCiDbygH07r'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Unzip
unzip -q ${filename}
rm ${filename}

# Download medbert-kd-chinese from Google Drive
filename='medbert-kd-chinese.zip'
fileid='18inyU0OPaPJLh7UleQh8g9hrQV4Og9Cv'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Unzip
unzip -q ${filename}
rm ${filename}

# Download bart-base-chinese from Google Drive
filename='bart-base-chinese.zip'
fileid='1W6Yu3-WBrDuxg9qGs1GDCtxzmhpaKEv6'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Unzip
unzip -q ${filename}
rm ${filename}

rm ./cookie
