---
layout: pages
title: Leetcode
index_img: /img/banner/algorithm.jpg
date: 2025-07-01 15:57:45
tags:
---

## 图论

### 带权并查集

* 两点是否联通，联通图上两点的路径

* 无向图判环

[399. 除法求值](https://leetcode.cn/problems/evaluate-division/)：给定若干等式 `Ai / Bi = values[i]`，每个 `Ai` 和 `Bi` 是字符串变量。还有一些查询 `Cj / Dj`，请根据已知等式计算每个查询的结果。

返回每个查询的结果，若无法确定某个结果，则返回 `-1.0`。

- 所有输入有效，除数不会为 0；
- 未出现在等式中的变量视为未定义，查询结果为 `-1.0`。



思路：若已知`a/b` 和 `b/c` 则可以推出`a/c`，  而这类问题就可以通过**带权并查集**解决

```c++
unordered_map<string, string>p;    //p[x]代表x的直接父节点
unordered_map<string, double>dis; //dis[x]代表x到p[x]的距离

/* 压缩路径
    a -> b -> c -> d （根节点）
    调用 find_root(a) 会触发：
    find_root(a)
        └── find_root(b)
            └── find_root(c)
                    └── find_root(d) // d 是根节点
    从最底层返回后，逐层更新路径压缩：
    p[c] = d
    p[b] = d
    p[a] = d
*/
string find_root(string x){
    if(!p.count(x)){
        p[x] = x;
        dis[x] = 1.0;
    }
    else{
        if(x != p[x]){
            string t = p[x];
            p[x] = find_root(p[x]);
            dis[x] *= dis[t];
        }
    }
    return p[x];
}
bool merge(string x, string y, double value){
    string fx = find_root(x);
    string fy = find_root(y);
    if(fx != fy){
        p[fx] = fy;
        dis[fx] = value*dis[y]/dis[x];
    }
    return true;
}
double isConnected(string x, string y){
    string fx = find_root(x);
    string fy = find_root(y);
    if(fx != fy){
        return -1.0;
    }
    else{
        return dis[x]/dis[y];
    }
}
```



### 拓扑排序

* 有向图判环

[207. 课程表](https://leetcode.cn/problems/course-schedule/description/?envType=study-plan-v2&envId=top-interview-150)：你需要修完 `numCourses` 门课程，课程编号为 `0 ~ numCourses - 1`。
 部分课程有先修要求，给定 `prerequisites[i] = [a, b]` 表示：修课程 `a` 前必须先修 `b`。请判断是否**能完成所有课程**（即是否不存在循环依赖）。 返回 `true` 表示可以完成，`false` 表示存在循环。

思路：经典有向图判环，先构建出图，如果存在拓扑排序则无环。
