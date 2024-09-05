conda install -c omgarcia gcc-6 # install GCC version 6
conda install libgcc            # install conda gcc tools

# make sure that you see GLIBCXX_3.4.xx on the list (which it could not find before)
strings /home/ywan1053/.conda/envs/cls/lib/libstdc++.so.6 | grep GLIBCXX

# add it to library paths

export LD_LIBRARY_PATH=/home/ywan1053/.conda/envs/cls/lib:$LD_LIBRARY_PATH