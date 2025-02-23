rm -rf compression_challenge_submission*
rm *.txt
ls -la
python compress.py
echo
read -n 1 -s -r -p "Press any key to continue"
echo
./evaluate.sh compression_challenge_submission.zip
