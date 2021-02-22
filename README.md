# Instruction
On Colab you have to change runtime to GPU

Then launch this command:
```
  !wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
  !dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
  !apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
  !apt-get update
  !apt-get install cuda-9.2
  ```

To verify if it works:
```
  !nvcc --version
```

Setup your google drive:
```
  from google.colab import drive
  drive.mount('/content/gdrive')
```

NOTE: Click on the link, choose you account, copy the code and then press enter

On Colab move to this directory
```
  cd /content/gdrive/MyDrive/
```

From Google Drive create a folder and copy dot.c file (my example file) in this folder.
Then on Colab move to the directory.

Verify if the file is present
```
  ls
```

Compile
```
  !nvcc dot.c -lcublas -o dot
```

Run
```
  !./dot
```
