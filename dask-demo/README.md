### Demonstration of active storage with `dask`

* The demonstration in this repository uses a modified version of
  dask: https://github.com/davidhassell/dask/tree/active-storage

* To install the modified version of `dask`:

```console
$ pip install git+ssh://git@github.com/davidhassell/dask.git@active-storage
...
Successfully installed dask-2022.4.1+38.geef967a8
```

* Code changes in the modified dask can be seen at
  https://github.com/davidhassell/dask/pull/1/files (66 newlines of
  code).

* The full results of running `demo.py` are shown here, and the dask
  graph visualisations are in the repository.

```console
$ cd dask-demo
$ python demo.py

Active max(a) = 0.146
Normal max(a) = 0.146

Active mean(a) = 0.046075
Normal mean(a) = 0.046075

Non-active sum(a) = 1.843
    Normal sum(a) = 1.843

Active max(a) + a = [[0.153 0.18  0.149 0.16  0.164 0.183 0.17  0.175]
 [0.169 0.182 0.191 0.208 0.192 0.219 0.152 0.212]
 [0.256 0.277 0.27  0.292 0.233 0.249 0.203 0.157]
 [0.175 0.205 0.185 0.216 0.204 0.218 0.155 0.163]
 [0.152 0.182 0.165 0.181 0.164 0.183 0.18  0.159]]
Normal max(a) + a = [[0.153 0.18  0.149 0.16  0.164 0.183 0.17  0.175]
 [0.169 0.182 0.191 0.208 0.192 0.219 0.152 0.212]
 [0.256 0.277 0.27  0.292 0.233 0.249 0.203 0.157]
 [0.175 0.205 0.185 0.216 0.204 0.218 0.155 0.163]
 [0.152 0.182 0.165 0.181 0.164 0.183 0.18  0.159]]

Active sum(max(a) + a) = 7.683
Normal sum(max(a) + a) = 7.683

$ # List of dask graph visualisations of active and normal operations
$ ls -1rt *.png
active_max.png
normal_max.png
active_mean.png
normal_mean.png
non_active_sum.png
normal_sum.png
active_max+a.png
normal_max+a.png
active_sum_max+a.png
normal_sum_max+a.png
```
