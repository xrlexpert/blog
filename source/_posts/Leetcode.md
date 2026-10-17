---
layout: pages
title: Leetcode
index_img: /img/banner/algorithm.jpg
date: 2025-07-01 15:57:45
tags:
---
## Intro
本文记录实习算法刷题，以及回顾本科期间的算法基础:) 希望能帮助大家速成

## 常用数据结构(C++)
`string s` 字符串

| API | 功能 | 示例 | 时间复杂度 |
|---|---|---|---|
| `s.size()` | 返回字符串长度 | `s.size()` | O(1) |
| `s.length()` | 返回字符串长度 | `s.length()` | O(1) |
| `s[i]` | 访问第 i 个字符 | `s[0]` | O(1) |
| `s.at(i)` | 访问第 i 个字符（越界检查） | `s.at(0)` | O(1) |
| `s.empty()` | 判断字符串是否为空 | `s.empty()` | O(1) |
| `s.clear()` | 清空字符串 | `s.clear()` | O(n) |
| `s.push_back(c)` | 尾部添加字符 | `s.push_back('a')` | 均摊 O(1) |
| `s.pop_back()` | 删除末尾字符 | `s.pop_back()` | O(1) |
| `s += str` | 拼接字符串 | `s += "abc"` | O(k) |
| `s.append(str)` | 追加字符串 | `s.append("abc")` | O(k) |
| `s.insert(pos,str)` | 插入字符串 | `s.insert(2,"abc")` | O(n) |
| `s.erase(pos,len)` | 删除指定区间字符 | `s.erase(2,3)` | O(n) |
| `s.replace(pos,len,str)` | 替换字符串 | `s.replace(1,2,"xx")` | O(n) |
| `s.find(str)` | 查找第一次出现位置 | `s.find("abc")` | O(nm) |
| `s.rfind(str)` | 从后向前查找 | `s.rfind("abc")` | O(nm) |
| `s.substr(pos,len)` | 截取子串 | `s.substr(2,3)` | O(len) |
| `s.compare(str)` | 字符串比较 | `s.compare("abc")` | O(min(n,m)) |
| `sort(s.begin(),s.end())` | 字符排序（升序） | `sort(s.begin(),s.end())` | O(nlogn) |
| `reverse(s.begin(),s.end())` | 反转字符串 | `reverse(s.begin(),s.end())` | O(n) |

`unorderd_map<T, T>mp;` 
* 基于哈希表，元素顺序无序

| API | 功能 | 平均时间复杂度 | 最坏时间复杂度 |
|---|---|---|---|
| `mp[key]` | 访问/插入元素 | O(1) | O(n) |
| `mp.insert({key,value})` | 插入键值对 | O(1) | O(n) |
| `mp.emplace(key,value)` | 原地插入 | O(1) | O(n) |
| `mp.find(key)` | 查找 key | O(1) | O(n) |
| `mp.count(key)` | 判断 key 是否存在 | O(1) | O(n) |
| `mp.erase(key)` | 根据 key 删除 | O(1) | O(n) |
| `mp.erase(iterator)` | 根据迭代器删除 | O(1) | O(1) |
| `mp.size()` | 返回元素数量 | O(1) | O(1) |
| `mp.empty()` | 判断是否为空 | O(1) | O(1) |
| `mp.clear()` | 清空所有元素 | O(n) | O(n) |
| `mp.begin()` | 返回头迭代器 | O(1) | O(1) |
| `mp.end()` | 返回尾迭代器 | O(1) | O(1) |
| 遍历 `for(auto p:mp)` | 遍历全部元素 | O(n) | O(n) |

例题：

[1.两数之和-简单](https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked)

[49. 字母异位词分组-中等](https://leetcode.cn/problems/group-anagrams/?envType=study-plan-v2&envId=top-100-liked)


`map<T, T>mp;` 
* 基于红黑树，元素顺序按照key从小到大排序

| API | 功能 | 平均时间复杂度 | 最坏时间复杂度 |
|---|---|---|---|
| `mp[key]` | 访问 / 插入元素 | O(log n) | O(log n) |
| `mp.insert({key,value})` | 插入键值对 | O(log n) | O(log n) |
| `mp.emplace(key,value)` | 原地插入 | O(log n) | O(log n) |
| `mp.find(key)` | 查找 key | O(log n) | O(log n) |
| `mp.count(key)` | 判断 key 是否存在 | O(log n) | O(log n) |
| `mp.erase(key)` | 根据 key 删除 | O(log n) | O(log n) |
| `mp.erase(iterator)` | 根据迭代器删除 | O(1)（均摊） | O(1) |
| `mp.size()` | 返回元素数量 | O(1) | O(1) |
| `mp.empty()` | 判断是否为空 | O(1) | O(1) |
| `mp.clear()` | 清空所有元素 | O(n) | O(n) |
| `mp.begin()` | 返回最小 key 的迭代器 | O(1) | O(1) |
| `mp.end()` | 返回尾后迭代器 | O(1) | O(1) |
| `mp.lower_bound(key)` | 查找第一个 ≥ key 的元素 | O(log n) | O(log n) |
| `mp.upper_bound(key)` | 查找第一个 > key 的元素 | O(log n) | O(log n) |
| `for(auto p:mp)` | 遍历全部元素 | O(n) | O(n) |

`unordered_set<int> st;` 以及 `set<int> st;`
| API | 功能 | 示例 | unordered_set复杂度 | set复杂度 |
|-|-|-|-|-|
| `insert(x)` | 插入元素 | `st.insert(5)` | 平均 O(1) | O(log n) |
| `emplace(x)` | 原地插入 | `st.emplace(5)` | 平均 O(1) | O(log n) |

例题：

[128. 最长连续序列-中等](https://leetcode.cn/problems/longest-consecutive-sequence/description/)

## 双指针

### 快慢指针
[283.移动零-简单](https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)：细节很多

<details>
<summary>实现细节</summary>

```python
void moveZeroes(vector<int>& nums) {
    int len = nums.size();
    int slow = 0;
    while(slow < len && nums[slow] != 0) slow++;
    for(int fast = slow + 1;fast < len;fast++){
        if(nums[fast] != 0){
            swap(nums[slow], nums[fast]);
            slow +=1;
        }
    }
    return ;
}
```

* `slow`是第一个零的位置，为什么+1后的位置就一定是0，因为fast>=slow+1且进行了交换操作，因此保证[slow+1, fast]区间都是连续的0。
* `fast`是非零的位置才需要交换，位于slow之后，因此外层使用for循环。
</details>

### 贪心指针
* 排序
* 水池体积

[15.三数之和-中等](https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked)： 此题细节很多，重点在于如何不重复不漏
<details>
<summary>实现细节</summary>

```python
vector<vector<int>> threeSum(vector<int>& nums) {
    int len = nums.size();
    //题目对于重复的定义其实是对其值的定义，而不是其位置，所以可以采用排序
    sort(nums.begin(), nums.end());
    vector<vector<int>> ans;
    for(int i = 0;i < len - 2; i++){
        if(i > 0 && nums[i] == nums[i-1]) continue; //已经检测过了才跳过
        int l = i + 1;
        int r = len - 1;
        int target = 0 - nums[i];
        while(l < r){
            if(nums[l] + nums[r] == target){
                ans.push_back({nums[i], nums[l], nums[r]});
                while(l < r && nums[l] == nums[l + 1]) l++; //已经加入了这组l，跳过
                while(l < r && nums[r] == nums[r - 1]) r--; //已经加入了这组r，跳过
                l++; 
                r--;
            }
            else if(nums[l] + nums[r] < target) l++;
            else r--;
        }
    }
}
```
* 注意不能先进行相同则跳过`while(i+1 < len && nums[i] == nums[i+1]) i++;`,因为可能出现[-1,-1,2]这种second==first的情况，先跳过的话会将这种三元组丢失掉。所以一定是先判断是否为三元组，之后才进行跳过的步骤
</details>

## DP
### 基础DP
[42.接雨水-困难](https://leetcode.cn/problems/trapping-rain-water/description/)

```python
class Solution {
public:
    int trap(vector<int>& height) {
        int len = height.size();
        int leftMAX = 0;
        int rightMAX = 0;
        int left = 0;
        int right = len - 1;
        int ans = 0;
        while(left < right){
            if(height[left] <= height[right]){
                ans += max(0,leftMAX - height[left]);
                leftMAX = max(leftMAX, height[left]);
                left ++;
            }
            else{
                ans += max(0, rightMAX - height[right]);
                rightMAX = max(rightMAX, height[right]);
                right --;
            }
        }
        return ans;
    }
};
```
* 因为这道题目两端最开始的位置对答案一定没有贡献，因此先判断`height[left] <= height[right]`再更新`leftMAX`和`rightMAX`，可以保证当`height[left] <= height[right]`时，`leftMAX`<=`rightMAX`一定成立，换句话说，MAX的值是在判断height之后才更新的，是跟着指针方向走的。
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
