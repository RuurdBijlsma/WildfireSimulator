this only ended up working on linux for me Ubuntu 20
1. Use proprietary nvidia drivers (additional drivers app)
2. Install cuda toolkit https://developer.nvidia.com/cuda-downloads
3. Install anaconda
4. Download boost https://www.boost.org/users/download/
   1. `./bootstrap.sh`
   2. Create `user-config.jam` in home dir 
   3. `sudo ./b2 runtime-link=static link=static install`
   4. Reload cmake config
   5. Build project

Now the `.so` file can be imported in python. 


`user-config.jam`:
```
using python 
    : 3.9                   # Version
    : /home/ruurd/anaconda3/bin/python      # Python Path
    : /home/ruurd/anaconda3/include/python3.9         # include path
    : /home/ruurd/anaconda3/lib            # lib path(s)
    : <define>BOOST_ALL_NO_LIB=1
    ;
```