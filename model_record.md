# 模型记录

`vgg16_cifar`:  

- $\alpha$=90%
(no psae)
```
--------------Compress Rate--------------
Channels Prune Rate: 2491/4224 (41.03%)
Params Compress Rate: 5.05 M/14.73 M(65.73%)
FLOPS Compress Rate: 143.56 M/314.03 M(54.28%)

--------------best code--------------
[9, 9, 9, 5, 5, 8, 7, 9, 3, 6, 9, 5, 1]

Best accurary: 93.180
```

- $\alpha$=80%
(no psae)
```
--------------Compress Rate--------------
Channels Prune Rate: 2088/4224 (50.57%)
Params Compress Rate: 3.65 M/14.73 M(75.20%)
FLOPS Compress Rate: 148.60 M/314.03 M(52.68%)

--------------best code--------------
[7, 8, 8, 8, 7, 7, 8, 8, 8, 1, 5, 1, 1]

Best accurary: 93.100
```


- $\alpha$ = 80%
(pase)

```
--------------Compress Rate--------------
Channels Prune Rate: 2401/4224 (43.16%)
Params Compress Rate: 4.50 M/14.73 M(69.46%)
FLOPS Compress Rate: 122.99 M/314.03 M(60.83%)
--------------best code--------------
[4, 8, 8, 8, 8, 8, 5, 7, 1, 7, 8, 7, 1]

 Best accurary: 93.300
```
