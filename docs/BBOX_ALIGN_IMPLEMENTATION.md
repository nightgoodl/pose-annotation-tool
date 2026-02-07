# Bbox Align 实现文档

## 概述

本文档描述了位姿标注工具中 **Bbox Align（边界框对齐）** 功能的实现原理和具体步骤。该功能确保变换后的 mesh 边与 GT bbox 边保持平行，同时支持各向异性缩放。

## 坐标系约定

| 坐标系 | 轴向定义 | 使用场景 |
|--------|----------|----------|
| **Mesh 坐标系** | Y-up（Three.js 默认） | 3D 模型渲染、localPoint |
| **世界坐标系** | Z-up | GT bbox、worldPoint、相机位姿 |
| **相机坐标系** | OpenCV 标准（+X右, +Y下, +Z前） | 深度反投影 |

### Y-up 到 Z-up 变换矩阵

```
R_yup_to_zup = | 1   0   0 |
               | 0   0  -1 |
               | 0   1   0 |
```

变换公式：
- x' = x
- y' = -z
- z' = y

## 核心算法

### 1. computeBboxAlignTransform（纯 Bbox 对齐）

**用途**：在没有点对的情况下，仅使用 bbox 信息进行初始对齐。

**变换公式**：
```
P_world = scale * (gtR * R_yup_to_zup) * P_mesh + t
```

**步骤**：

1. **获取 mesh 信息**：考虑 `MESH_SCALE = 2.0` 的缩放
   ```typescript
   scaledMeshCenter = meshInfo.center * MESH_SCALE
   scaledMeshExtent = meshInfo.extent * MESH_SCALE
   ```

2. **计算 GT bbox 尺寸**：优先从 corners 计算
   ```typescript
   gtExtent = [max - min] for each axis
   gtCenter = (max + min) / 2
   ```

3. **计算各向同性 scale**：使用对角线比例
   ```typescript
   meshDiagonal = sqrt(extent_x² + extent_y² + extent_z²)
   gtDiagonal = sqrt(gtExtent_x² + gtExtent_y² + gtExtent_z²)
   scale = gtDiagonal / meshDiagonal
   ```

4. **构建组合旋转矩阵**：
   ```typescript
   combinedR = gtR * R_yup_to_zup
   ```

5. **计算 translation**：让 mesh 中心对齐 GT bbox 中心
   ```typescript
   t = gtCenter - scale * combinedR * meshCenter
   ```

6. **构建变换矩阵**：
   ```typescript
   transformMatrix = [scale * combinedR | t]
   ```

### 2. solveUmeyamaConstrained（带约束的 Umeyama 算法）

**用途**：使用用户标注的点对，求解最优的旋转、缩放和平移。

**约束**：最终旋转必须是 `baseRotation * rotZ(θ)` 的形式，其中 θ ∈ {0°, 90°, 180°, 270°}。

**变换公式**：
```
P_world = R * S * R_yup_to_zup * P_mesh + t
```

其中：
- `R = baseRotation * rotZ(θ)` — 组合旋转矩阵
- `S = diag(sx, sy, sz)` — 各向异性缩放矩阵
- `R_yup_to_zup` — Y-up 到 Z-up 变换

**步骤**：

1. **预处理 srcPoints**：将 localPoint 从 Y-up 变换到 Z-up
   ```typescript
   srcPoints_zup = {
     x: localPoint.x,
     y: -localPoint.z,
     z: localPoint.y
   }
   ```

2. **测试四个旋转角度**：
   ```typescript
   for (angle in [0°, 90°, 180°, 270°]) {
     rotZ = rotationMatrixZ(angle)
     R_test = baseRotation * rotZ
     result = solveScaleTranslationForFixedR(srcPoints_zup, dstPoints, R_test)
     // 选择误差最小的结果
   }
   ```

3. **求解 scale 和 translation**（在 solveScaleTranslationForFixedR 中）：
   - 计算质心并去中心化
   - 将点变换到 R 的局部坐标系
   - 分别计算各轴的 scale
   - 计算 translation

4. **构建最终变换矩阵**：
   ```typescript
   // result.transformMatrix 是针对 Z-up 坐标的
   // 需要乘以 R_yup_to_zup 来应用到 Y-up 的 mesh
   finalTransform = result.transformMatrix * R_yup_to_zup
   ```

### 3. solveScaleTranslationForFixedR（固定旋转求解 scale 和 translation）

**变换公式**：
```
dst = R * diag(sx, sy, sz) * src + t
```

**求解方法**：

1. **计算质心**：
   ```typescript
   xCentroid = mean(srcPoints)
   yCentroid = mean(dstPoints)
   ```

2. **去中心化**：
   ```typescript
   srcCentered = srcPoints - xCentroid
   dstCentered = dstPoints - yCentroid
   ```

3. **变换到 R 的局部坐标系**：
   ```typescript
   // 将 dstCentered 变换到 R 的局部坐标系
   y_local = R^T * dstCentered
   ```

4. **分别计算各轴 scale**：
   ```typescript
   for (axis in [0, 1, 2]) {
     numerator = Σ(y_local[axis] * srcCentered[axis])
     denominator = Σ(srcCentered[axis]²)
     scaleVec[axis] = numerator / denominator
   }
   ```

5. **计算 translation**：
   ```typescript
   // 在局部坐标系中计算
   t_local = yCentroid_local - S * xCentroid
   // 变换回世界坐标系
   t = R * t_local
   ```

6. **构建变换矩阵**：
   ```typescript
   RS = R * diag(scaleVec)
   transformMatrix = [RS | t]
   ```

## 关键修复

### 修复 1：旋转矩阵乘法顺序

**问题**：`R_test = rotZ * baseRotation`（错误）

**修复**：`R_test = baseRotation * rotZ`（正确）

**原因**：矩阵乘法顺序表示变换的应用顺序。正确的顺序是先绕 mesh 的 Z 轴旋转（选择 0°/90°/180°/270°），再应用 GT bbox 的旋转。

### 修复 2：Y-up 到 Z-up 坐标变换

**问题**：mesh 是 Y-up 的，而世界坐标系是 Z-up 的，直接使用 localPoint 会导致：
- scale 计算错误（轴向不匹配）
- 片状物体"躺着"（Y 和 Z 混淆）

**修复**：
1. 在 `runAlignment` 中，将 srcPoints 从 Y-up 变换到 Z-up
2. 在最终变换矩阵中包含 `R_yup_to_zup`
3. 在 `computeBboxAlignTransform` 中也添加 `R_yup_to_zup`

## 文件修改列表

| 文件 | 函数/位置 | 修改内容 |
|------|-----------|----------|
| `src/utils/math.ts` | `solveUmeyamaConstrained` | 修复 `R_test = baseRotation * rotZ` |
| `src/utils/math.ts` | `computeBboxAlignTransform` | 添加 `R_yup_to_zup` 变换 |
| `src/stores/mvAnnotationStore.ts` | `runAlignment` | srcPoints Y-up→Z-up 变换，最终矩阵乘以 `R_yup_to_zup` |

## 调试信息

运行时会输出以下调试信息：

```
[runAlignment] Point pairs (after Y-up to Z-up transform):
  Pair 0: local_orig=(x, y, z) -> local_zup=(x', y', z') -> world=(...)

[computeBboxAlignTransform] combinedR (gtR * R_yup_to_zup): [...]
[computeBboxAlignTransform] translationCombined: [...]

[Constrained Umeyama] baseRotation: [...]
[Constrained Umeyama] angle=0°, scaleVec=[sx, sy, sz], error=...

[solveScaleTranslationForFixedR] scaleVec=[sx, sy, sz], error=...
```

## 使用流程

1. **初始对齐**：点击 "Bbox Align" 按钮，使用 `computeBboxAlignTransform` 进行初始对齐
2. **精细调整**：在图像上点击 bbox 角点，在 mesh 上点击对应点，建立点对
3. **自动对齐**：当点对数量 ≥ 3 时，自动调用 `runAlignment` 进行对齐
4. **RANSAC 对齐**：当点对数量 ≥ 5 时，可手动触发 RANSAC 对齐以剔除离群点

## 数学保证

1. **边边平行**：通过使用 GT bbox 的旋转矩阵 R，保证变换后的 mesh 边与 GT bbox 边平行
2. **离散旋转**：只测试 4 个离散角度（0°, 90°, 180°, 270°），符合物体对称性假设
3. **各向异性缩放**：在 R 的局部坐标系中分别计算各轴 scale，实现各向异性缩放
4. **坐标系一致**：通过 Y-up 到 Z-up 变换，确保 mesh 坐标系和世界坐标系一致
