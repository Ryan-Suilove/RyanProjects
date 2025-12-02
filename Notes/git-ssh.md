# 🔑 GitHub SSH 免密配置记录（2025-12-02）

## 1. 📌 目的

首次创建 GitHub 仓库并完成本机与 GitHub 的 **SSH 免密登录配置**，用于后续管理个人项目仓库 **RyanProjects**（克隆、提交、推送等全部免密）。

---

## 2. 🛠️ 操作步骤记录

### **2.1 生成 SSH Key**

在本地终端运行：

```bash
ssh-keygen -t ed25519 -C "1969608958@qq.com"
```

生成内容包括：

* **私钥**：`~/.ssh/id_ed25519`
* **公钥**：`~/.ssh/id_ed25519.pub`

---

### **2.2 添加公钥到 GitHub**

GitHub 网站路径：
**Settings → SSH and GPG Keys → New SSH Key**
将公钥文件内容粘贴进去即可。

---

### **2.3 测试 SSH 连接**

```bash
ssh -T git@github.com
```

成功输出：

```
Hi Ryan-Suilove! You've successfully authenticated, but GitHub does not provide shell access.
```

---

### **2.4 配置 ssh-agent（用于保存私钥口令）**

以管理员方式启动 PowerShell，并执行：

```powershell
Set-Service -Name ssh-agent -StartupType Automatic
Start-Service ssh-agent
```

将私钥加入 agent（只需一次）：

```bash
ssh-add ~/.ssh/id_ed25519
```

---

### **2.5 使用 SSH 克隆仓库**

```bash
git clone git@github.com:Ryan-Suilove/RyanProjects.git
```

克隆成功后，以后所有：

* `git pull`
* `git push`
* `git fetch`

都不需要再输入密码或口令。

---

## 3. 🧩 遇到的问题 & 解决方案

### ❗ ssh-agent 无法启动

报错：无法启动服务
**原因：** ssh-agent 服务默认是 Disabled（禁用状态）
**解决：** 用管理员权限运行 PowerShell 并手动启用 StartupType → Automatic

---

## 4. 📚 学到的内容总结

### ✔ SSH key 的作用

* **私钥（本地）**：相当于你的身份
* **公钥（上传 GitHub）**：GitHub 用它验证你的身份
* 以后不再需要输入 GitHub 密码

### ✔ SSH 免密的原理

GitHub 通过公钥验证你本地的私钥是否匹配 → 安全且免密码
HTTPS 永远会要求密码，但 SSH 不会。

### ✔ ssh-agent 作用

* 存储私钥口令（passphrase）
* 开机自动加载后，无需重复输入

---

## 5. 🏁 最终结果

* 本机已成功配置 **GitHub SSH 免密登录**
* 支持 git clone / push / pull 全程免密码
* RyanProjects 仓库已成功克隆到本地
* 今后个人项目可通过 Git 统一管理

---

# 📘 附：常用 Markdown 排版技巧（简短但实用）

### ✔ 代码块（可复制区域）

```bash
命令内容
```

### ✔ 行内代码

使用：`代码`
例：`git status`

### ✔ 标题层级

```
# 一级标题  
## 二级标题  
### 三级标题
```

### ✔ 加粗、斜体

```
**加粗**  
*斜体*
```

### ✔ 列表

```
- 项目1  
- 项目2  

1. 有序1  
2. 有序2
```

### ✔ 分隔线

```
---
```

---

