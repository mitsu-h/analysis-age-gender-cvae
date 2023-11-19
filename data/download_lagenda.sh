curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1QXO0NlkABPZT6x1_0Uc2i6KAtdcrpTbG" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1QXO0NlkABPZT6x1_0Uc2i6KAtdcrpTbG" -o lagenda_images.zip
