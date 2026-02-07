/**
 * 数学工具函数 - 矩阵运算、Umeyama算法、反投影
 * 
 * 坐标系约定 (基于 COORDINATE_SYSTEM_SUMMARY.md):
 * - 相机坐标系: OpenCV标准, +X右, +Y下, +Z前
 * - 世界坐标系: Z-up
 * - RT矩阵: camera-to-world (P_world = R @ P_cam + t)
 */

import type { Matrix3, Matrix4, Vector3, Point3D, CameraIntrinsics, UmeyamaResult } from '../types';

// ============== 基础矩阵运算 ==============

/**
 * 创建单位矩阵 4x4
 */
export function identity4(): Matrix4 {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
}

/**
 * 创建单位矩阵 3x3
 */
export function identity3(): Matrix3 {
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ];
}

/**
 * 4x4 矩阵乘法
 */
export function multiply4x4(a: Matrix4, b: Matrix4): Matrix4 {
  const result: Matrix4 = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
  ];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      for (let k = 0; k < 4; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
}

/**
 * 3x3 矩阵乘法
 */
export function multiply3x3(a: Matrix3, b: Matrix3): Matrix3 {
  const result: Matrix3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
}

/**
 * 3x3 矩阵乘向量
 */
export function multiplyMat3Vec3(m: Matrix3, v: Vector3): Vector3 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]
  ];
}

/**
 * 4x4 矩阵乘向量 (齐次坐标)
 */
export function multiplyMat4Vec4(m: Matrix4, v: [number, number, number, number]): [number, number, number, number] {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
    m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3]
  ];
}

/**
 * 4x4 矩阵变换 3D 点 (自动处理齐次坐标)
 */
export function transformPoint(m: Matrix4, p: Point3D): Point3D {
  const result = multiplyMat4Vec4(m, [p.x, p.y, p.z, 1]);
  return {
    x: result[0] / result[3],
    y: result[1] / result[3],
    z: result[2] / result[3]
  };
}

/**
 * 3x3 矩阵转置
 */
export function transpose3(m: Matrix3): Matrix3 {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]]
  ];
}

/**
 * 4x4 矩阵转置
 */
export function transpose4(m: Matrix4): Matrix4 {
  return [
    [m[0][0], m[1][0], m[2][0], m[3][0]],
    [m[0][1], m[1][1], m[2][1], m[3][1]],
    [m[0][2], m[1][2], m[2][2], m[3][2]],
    [m[0][3], m[1][3], m[2][3], m[3][3]]
  ];
}

/**
 * 3x3 矩阵行列式
 */
export function determinant3(m: Matrix3): number {
  return (
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  );
}

/**
 * 4x4 矩阵求逆
 */
export function inverse4(m: Matrix4): Matrix4 {
  const a = m[0][0], b = m[0][1], c = m[0][2], d = m[0][3];
  const e = m[1][0], f = m[1][1], g = m[1][2], h = m[1][3];
  const i = m[2][0], j = m[2][1], k = m[2][2], l = m[2][3];
  const n = m[3][0], o = m[3][1], p = m[3][2], q = m[3][3];

  const kq_lp = k * q - l * p;
  const jq_lo = j * q - l * o;
  const jp_ko = j * p - k * o;
  const iq_ln = i * q - l * n;
  const ip_kn = i * p - k * n;
  const io_jn = i * o - j * n;

  const det = a * (f * kq_lp - g * jq_lo + h * jp_ko)
            - b * (e * kq_lp - g * iq_ln + h * ip_kn)
            + c * (e * jq_lo - f * iq_ln + h * io_jn)
            - d * (e * jp_ko - f * ip_kn + g * io_jn);

  if (Math.abs(det) < 1e-10) {
    console.warn('Matrix is singular, returning identity');
    return identity4();
  }

  const invDet = 1 / det;

  const gq_hp = g * q - h * p;
  const fq_ho = f * q - h * o;
  const fp_go = f * p - g * o;
  const eq_hn = e * q - h * n;
  const ep_gn = e * p - g * n;
  const eo_fn = e * o - f * n;
  const gl_hk = g * l - h * k;
  const fl_hj = f * l - h * j;
  const fk_gj = f * k - g * j;
  const el_hi = e * l - h * i;
  const ek_gi = e * k - g * i;
  const ej_fi = e * j - f * i;

  return [
    [
      (f * kq_lp - g * jq_lo + h * jp_ko) * invDet,
      (-b * kq_lp + c * jq_lo - d * jp_ko) * invDet,
      (b * gq_hp - c * fq_ho + d * fp_go) * invDet,
      (-b * gl_hk + c * fl_hj - d * fk_gj) * invDet
    ],
    [
      (-e * kq_lp + g * iq_ln - h * ip_kn) * invDet,
      (a * kq_lp - c * iq_ln + d * ip_kn) * invDet,
      (-a * gq_hp + c * eq_hn - d * ep_gn) * invDet,
      (a * gl_hk - c * el_hi + d * ek_gi) * invDet
    ],
    [
      (e * jq_lo - f * iq_ln + h * io_jn) * invDet,
      (-a * jq_lo + b * iq_ln - d * io_jn) * invDet,
      (a * fq_ho - b * eq_hn + d * eo_fn) * invDet,
      (-a * fl_hj + b * el_hi - d * ej_fi) * invDet
    ],
    [
      (-e * jp_ko + f * ip_kn - g * io_jn) * invDet,
      (a * jp_ko - b * ip_kn + c * io_jn) * invDet,
      (-a * fp_go + b * ep_gn - c * eo_fn) * invDet,
      (a * fk_gj - b * ek_gi + c * ej_fi) * invDet
    ]
  ];
}

/**
 * 3x3 矩阵求逆
 */
export function inverse3(m: Matrix3): Matrix3 {
  const det = determinant3(m);
  if (Math.abs(det) < 1e-10) {
    console.warn('Matrix is singular, returning identity');
    return identity3();
  }

  const invDet = 1 / det;
  return [
    [
      (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet,
      (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet,
      (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet
    ],
    [
      (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet,
      (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet,
      (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet
    ],
    [
      (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet,
      (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet,
      (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet
    ]
  ];
}

// ============== 向量运算 ==============

export function vec3Add(a: Vector3, b: Vector3): Vector3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function vec3Sub(a: Vector3, b: Vector3): Vector3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vec3Scale(v: Vector3, s: number): Vector3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

export function vec3Dot(a: Vector3, b: Vector3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function vec3Cross(a: Vector3, b: Vector3): Vector3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

export function vec3Length(v: Vector3): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

export function vec3Normalize(v: Vector3): Vector3 {
  const len = vec3Length(v);
  if (len < 1e-10) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

export function point3DToVec3(p: Point3D): Vector3 {
  return [p.x, p.y, p.z];
}

export function vec3ToPoint3D(v: Vector3): Point3D {
  return { x: v[0], y: v[1], z: v[2] };
}

// ============== SVD 分解 (Jacobi方法) ==============

/**
 * 简化的 3x3 SVD 分解 (Jacobi迭代法)
 * 返回 U, S, V 使得 A = U * diag(S) * V^T
 */
export function svd3x3(A: Matrix3): { U: Matrix3; S: Vector3; V: Matrix3 } {
  // 使用 A^T * A 的特征值分解来计算 SVD
  const AtA: Matrix3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        AtA[i][j] += A[k][i] * A[k][j];
      }
    }
  }

  // Jacobi 迭代求 AtA 的特征值和特征向量
  const { eigenvalues, eigenvectors } = jacobiEigen3x3(AtA);
  
  // V = eigenvectors (按特征值降序排列)
  const indices = [0, 1, 2].sort((a, b) => eigenvalues[b] - eigenvalues[a]);
  const V: Matrix3 = [
    [eigenvectors[0][indices[0]], eigenvectors[0][indices[1]], eigenvectors[0][indices[2]]],
    [eigenvectors[1][indices[0]], eigenvectors[1][indices[1]], eigenvectors[1][indices[2]]],
    [eigenvectors[2][indices[0]], eigenvectors[2][indices[1]], eigenvectors[2][indices[2]]]
  ];
  
  // S = sqrt(eigenvalues)
  const S: Vector3 = [
    Math.sqrt(Math.max(0, eigenvalues[indices[0]])),
    Math.sqrt(Math.max(0, eigenvalues[indices[1]])),
    Math.sqrt(Math.max(0, eigenvalues[indices[2]]))
  ];
  
  // U = A * V * S^(-1)
  const AV = multiply3x3(A, V);
  const U: Matrix3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      if (S[j] > 1e-10) {
        U[i][j] = AV[i][j] / S[j];
      }
    }
  }
  
  // 确保 U 是正交的
  orthonormalize3x3(U);
  
  return { U, S, V };
}

/**
 * 3x3 对称矩阵的 Jacobi 特征值分解
 */
function jacobiEigen3x3(A: Matrix3): { eigenvalues: Vector3; eigenvectors: Matrix3 } {
  const maxIter = 50;
  const tol = 1e-10;
  
  // 复制矩阵
  const D: Matrix3 = [
    [A[0][0], A[0][1], A[0][2]],
    [A[1][0], A[1][1], A[1][2]],
    [A[2][0], A[2][1], A[2][2]]
  ];
  
  // 初始化特征向量为单位矩阵
  const V: Matrix3 = identity3();
  
  for (let iter = 0; iter < maxIter; iter++) {
    // 找最大非对角元素
    let maxVal = 0;
    let p = 0, q = 1;
    
    for (let i = 0; i < 3; i++) {
      for (let j = i + 1; j < 3; j++) {
        if (Math.abs(D[i][j]) > maxVal) {
          maxVal = Math.abs(D[i][j]);
          p = i;
          q = j;
        }
      }
    }
    
    if (maxVal < tol) break;
    
    // 计算旋转角度
    const theta = (D[q][q] - D[p][p]) / (2 * D[p][q]);
    const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;
    
    // 更新 D
    const Dpp = D[p][p];
    const Dqq = D[q][q];
    const Dpq = D[p][q];
    
    D[p][p] = c * c * Dpp - 2 * s * c * Dpq + s * s * Dqq;
    D[q][q] = s * s * Dpp + 2 * s * c * Dpq + c * c * Dqq;
    D[p][q] = D[q][p] = 0;
    
    for (let k = 0; k < 3; k++) {
      if (k !== p && k !== q) {
        const Dkp = D[k][p];
        const Dkq = D[k][q];
        D[k][p] = D[p][k] = c * Dkp - s * Dkq;
        D[k][q] = D[q][k] = s * Dkp + c * Dkq;
      }
    }
    
    // 更新特征向量
    for (let k = 0; k < 3; k++) {
      const Vkp = V[k][p];
      const Vkq = V[k][q];
      V[k][p] = c * Vkp - s * Vkq;
      V[k][q] = s * Vkp + c * Vkq;
    }
  }
  
  return {
    eigenvalues: [D[0][0], D[1][1], D[2][2]],
    eigenvectors: V
  };
}

/**
 * 正交化 3x3 矩阵 (Gram-Schmidt)
 */
function orthonormalize3x3(M: Matrix3): void {
  // 第一列
  let len = Math.sqrt(M[0][0] * M[0][0] + M[1][0] * M[1][0] + M[2][0] * M[2][0]);
  if (len > 1e-10) {
    M[0][0] /= len;
    M[1][0] /= len;
    M[2][0] /= len;
  }
  
  // 第二列
  let dot = M[0][0] * M[0][1] + M[1][0] * M[1][1] + M[2][0] * M[2][1];
  M[0][1] -= dot * M[0][0];
  M[1][1] -= dot * M[1][0];
  M[2][1] -= dot * M[2][0];
  
  len = Math.sqrt(M[0][1] * M[0][1] + M[1][1] * M[1][1] + M[2][1] * M[2][1]);
  if (len > 1e-10) {
    M[0][1] /= len;
    M[1][1] /= len;
    M[2][1] /= len;
  }
  
  // 第三列 = 第一列 × 第二列
  M[0][2] = M[1][0] * M[2][1] - M[2][0] * M[1][1];
  M[1][2] = M[2][0] * M[0][1] - M[0][0] * M[2][1];
  M[2][2] = M[0][0] * M[1][1] - M[1][0] * M[0][1];
}

// ============== Umeyama 算法 ==============

/**
 * Umeyama 算法 - 带尺度估计的点集配准
 * 
 * 求解 s*R*src + t ≈ dst
 * 
 * @param srcPoints 源点集 (模型空间局部坐标)
 * @param dstPoints 目标点集 (世界空间坐标)
 * @returns 变换结果 { rotation, translation, scale, transformMatrix, error }
 */
export function solveUmeyama(srcPoints: Point3D[], dstPoints: Point3D[]): UmeyamaResult {
  const n = srcPoints.length;
  
  if (n < 3) {
    console.error('Umeyama requires at least 3 point pairs');
    return {
      rotation: identity3(),
      translation: [0, 0, 0],
      scale: 1,
      transformMatrix: identity4(),
      error: Infinity
    };
  }
  
  // 1. 计算质心
  const srcCentroid: Vector3 = [0, 0, 0];
  const dstCentroid: Vector3 = [0, 0, 0];
  
  for (let i = 0; i < n; i++) {
    srcCentroid[0] += srcPoints[i].x;
    srcCentroid[1] += srcPoints[i].y;
    srcCentroid[2] += srcPoints[i].z;
    dstCentroid[0] += dstPoints[i].x;
    dstCentroid[1] += dstPoints[i].y;
    dstCentroid[2] += dstPoints[i].z;
  }
  
  srcCentroid[0] /= n;
  srcCentroid[1] /= n;
  srcCentroid[2] /= n;
  dstCentroid[0] /= n;
  dstCentroid[1] /= n;
  dstCentroid[2] /= n;
  
  // 2. 去中心化
  const srcCentered: Vector3[] = [];
  const dstCentered: Vector3[] = [];
  
  for (let i = 0; i < n; i++) {
    srcCentered.push([
      srcPoints[i].x - srcCentroid[0],
      srcPoints[i].y - srcCentroid[1],
      srcPoints[i].z - srcCentroid[2]
    ]);
    dstCentered.push([
      dstPoints[i].x - dstCentroid[0],
      dstPoints[i].y - dstCentroid[1],
      dstPoints[i].z - dstCentroid[2]
    ]);
  }
  
  // 3. 计算方差
  let srcVariance = 0;
  for (let i = 0; i < n; i++) {
    srcVariance += vec3Dot(srcCentered[i], srcCentered[i]);
  }
  srcVariance /= n;
  
  // 4. 计算协方差矩阵 H = sum(dst * src^T) / n
  const H: Matrix3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ];
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        H[j][k] += dstCentered[i][j] * srcCentered[i][k];
      }
    }
  }
  
  for (let j = 0; j < 3; j++) {
    for (let k = 0; k < 3; k++) {
      H[j][k] /= n;
    }
  }
  
  // 5. SVD 分解 H = U * S * V^T
  const { U, S, V } = svd3x3(H);
  
  // 6. 计算旋转矩阵 R = U * V^T
  const Vt = transpose3(V);
  let R = multiply3x3(U, Vt);
  
  // 7. 确保 R 是旋转矩阵 (det(R) = 1)
  const detR = determinant3(R);
  if (detR < 0) {
    // 翻转 U 的最后一列
    U[0][2] = -U[0][2];
    U[1][2] = -U[1][2];
    U[2][2] = -U[2][2];
    R = multiply3x3(U, Vt);
  }
  
  // 8. 计算尺度 s = trace(S) / srcVariance
  const traceS = S[0] + S[1] + S[2];
  const scale = srcVariance > 1e-10 ? traceS / srcVariance : 1;
  
  // 9. 计算平移 t = dstCentroid - s * R * srcCentroid
  const RSrc = multiplyMat3Vec3(R, srcCentroid);
  const translation: Vector3 = [
    dstCentroid[0] - scale * RSrc[0],
    dstCentroid[1] - scale * RSrc[1],
    dstCentroid[2] - scale * RSrc[2]
  ];
  
  // 10. 构建完整的 4x4 变换矩阵
  const transformMatrix: Matrix4 = [
    [scale * R[0][0], scale * R[0][1], scale * R[0][2], translation[0]],
    [scale * R[1][0], scale * R[1][1], scale * R[1][2], translation[1]],
    [scale * R[2][0], scale * R[2][1], scale * R[2][2], translation[2]],
    [0, 0, 0, 1]
  ];
  
  // 11. 计算配准误差
  let error = 0;
  for (let i = 0; i < n; i++) {
    const transformed = transformPoint(transformMatrix, srcPoints[i]);
    const dx = transformed.x - dstPoints[i].x;
    const dy = transformed.y - dstPoints[i].y;
    const dz = transformed.z - dstPoints[i].z;
    error += dx * dx + dy * dy + dz * dz;
  }
  error = Math.sqrt(error / n);
  
  return {
    rotation: R,
    translation,
    scale,
    transformMatrix,
    error
  };
}

/**
 * RANSAC + Umeyama 算法 - 鲁棒的点集配准
 * 
 * 使用 RANSAC 剔除离群点，然后用内点重新计算变换
 * 
 * @param srcPoints 源点集 (模型空间局部坐标)
 * @param dstPoints 目标点集 (世界空间坐标)
 * @param options RANSAC 参数
 * @returns 变换结果 + 内点索引
 */
export function solveUmeyamaRANSAC(
  srcPoints: Point3D[],
  dstPoints: Point3D[],
  options: {
    maxIterations?: number;    // 最大迭代次数
    inlierThreshold?: number;  // 内点阈值（误差小于此值视为内点）
    minInlierRatio?: number;   // 最小内点比例
  } = {}
): UmeyamaResult & { inlierIndices: number[]; outlierIndices: number[] } {
  const n = srcPoints.length;
  const maxIterations = options.maxIterations ?? 100;
  const inlierThreshold = options.inlierThreshold ?? 0.05;  // 5cm 默认阈值
  const minInlierRatio = options.minInlierRatio ?? 0.5;
  
  // 至少需要 3 个点
  if (n < 3) {
    console.error('RANSAC requires at least 3 point pairs');
    return {
      ...solveUmeyama(srcPoints, dstPoints),
      inlierIndices: [],
      outlierIndices: Array.from({ length: n }, (_, i) => i)
    };
  }
  
  // 如果点数少于 4，直接使用所有点
  if (n <= 4) {
    const result = solveUmeyama(srcPoints, dstPoints);
    return {
      ...result,
      inlierIndices: Array.from({ length: n }, (_, i) => i),
      outlierIndices: []
    };
  }
  
  let bestResult: UmeyamaResult | null = null;
  let bestInliers: number[] = [];
  let bestError = Infinity;
  
  // RANSAC 迭代
  for (let iter = 0; iter < maxIterations; iter++) {
    // 1. 随机选择 3 个点（最小样本）
    const sampleIndices = randomSample(n, 3);
    const sampleSrc = sampleIndices.map(i => srcPoints[i]);
    const sampleDst = sampleIndices.map(i => dstPoints[i]);
    
    // 2. 用样本点计算变换
    const candidateResult = solveUmeyama(sampleSrc, sampleDst);
    
    // 3. 计算所有点的误差，找内点
    const inliers: number[] = [];
    let totalError = 0;
    
    for (let i = 0; i < n; i++) {
      const transformed = transformPoint(candidateResult.transformMatrix, srcPoints[i]);
      const dx = transformed.x - dstPoints[i].x;
      const dy = transformed.y - dstPoints[i].y;
      const dz = transformed.z - dstPoints[i].z;
      const error = Math.sqrt(dx * dx + dy * dy + dz * dz);
      
      if (error < inlierThreshold) {
        inliers.push(i);
        totalError += error;
      }
    }
    
    // 4. 如果内点数量更多，更新最佳结果
    if (inliers.length > bestInliers.length || 
        (inliers.length === bestInliers.length && totalError < bestError)) {
      bestInliers = inliers;
      bestError = totalError;
      bestResult = candidateResult;
    }
    
    // 5. 如果内点比例足够高，提前终止
    if (inliers.length / n >= 0.9) {
      break;
    }
  }
  
  // 6. 用所有内点重新计算最终变换
  if (bestInliers.length >= 3) {
    const inlierSrc = bestInliers.map(i => srcPoints[i]);
    const inlierDst = bestInliers.map(i => dstPoints[i]);
    bestResult = solveUmeyama(inlierSrc, inlierDst);
  }
  
  // 7. 计算离群点
  const outlierIndices = Array.from({ length: n }, (_, i) => i)
    .filter(i => !bestInliers.includes(i));
  
  console.log(`[RANSAC] 内点: ${bestInliers.length}/${n}, 离群点: ${outlierIndices.length}`);
  
  return {
    ...(bestResult ?? solveUmeyama(srcPoints, dstPoints)),
    inlierIndices: bestInliers,
    outlierIndices
  };
}

/**
 * 随机采样不重复的索引
 */
function randomSample(n: number, k: number): number[] {
  const indices: number[] = [];
  const used = new Set<number>();
  
  while (indices.length < k) {
    const idx = Math.floor(Math.random() * n);
    if (!used.has(idx)) {
      used.add(idx);
      indices.push(idx);
    }
  }
  
  return indices;
}

// ============== 反投影函数 ==============

/**
 * 将像素坐标反投影到世界坐标系
 * 
 * 基于 COORDINATE_SYSTEM_SUMMARY.md:
 * - K: 相机内参
 * - RT (camera-to-world): P_world = R @ P_cam + t
 * 
 * @param u 像素 x 坐标
 * @param v 像素 y 坐标
 * @param depth 深度值 (相机坐标系下的 Z 值)
 * @param K 相机内参
 * @param RT_c2w camera-to-world 变换矩阵 (4x4)
 * @returns 世界坐标系中的 3D 点
 */
export function unprojectPoint(
  u: number,
  v: number,
  depth: number,
  K: CameraIntrinsics,
  RT_c2w: Matrix4
): Point3D {
  // 1. 像素坐标 -> 归一化相机坐标 -> 相机坐标
  const x_cam = (u - K.cx) * depth / K.fx;
  const y_cam = (v - K.cy) * depth / K.fy;
  const z_cam = depth;
  
  // P_cam = [x_cam, y_cam, z_cam]
  const P_cam: Point3D = { x: x_cam, y: y_cam, z: z_cam };
  
  // 2. 相机坐标 -> 世界坐标
  // RT_c2w 是 camera-to-world: P_world = R @ P_cam + t
  // RT_c2w = [R | t], 直接使用 4x4 矩阵变换
  const P_world = transformPoint(RT_c2w, P_cam);
  
  return P_world;
}

/**
 * 将世界坐标投影到像素坐标
 * 
 * @param point 世界坐标系中的 3D 点
 * @param K 相机内参
 * @param RT_c2w camera-to-world 变换矩阵 (4x4)
 * @returns 像素坐标 {u, v} 和深度 depth，如果点在相机后方返回 null
 */
export function projectPoint(
  point: Point3D,
  K: CameraIntrinsics,
  RT_c2w: Matrix4
): { u: number; v: number; depth: number } | null {
  // 1. 世界坐标 -> 相机坐标
  // 需要 RT_c2w 的逆矩阵 (world-to-camera)
  const RT_w2c = inverse4(RT_c2w);
  const P_cam = transformPoint(RT_w2c, point);
  
  // 2. 检查是否在相机前方
  if (P_cam.z <= 0) {
    return null;
  }
  
  // 3. 相机坐标 -> 像素坐标
  const u = K.fx * P_cam.x / P_cam.z + K.cx;
  const v = K.fy * P_cam.y / P_cam.z + K.cy;
  
  return { u, v, depth: P_cam.z };
}

// ============== 辅助函数 ==============

/**
 * Matrix4 转为 Three.js 格式的 16 元素数组 (列优先)
 */
export function matrix4ToArray(m: Matrix4): number[] {
  return [
    m[0][0], m[1][0], m[2][0], m[3][0],
    m[0][1], m[1][1], m[2][1], m[3][1],
    m[0][2], m[1][2], m[2][2], m[3][2],
    m[0][3], m[1][3], m[2][3], m[3][3]
  ];
}

/**
 * 从 Three.js 格式的 16 元素数组 (列优先) 转为 Matrix4
 */
export function arrayToMatrix4(arr: number[]): Matrix4 {
  return [
    [arr[0], arr[4], arr[8], arr[12]],
    [arr[1], arr[5], arr[9], arr[13]],
    [arr[2], arr[6], arr[10], arr[14]],
    [arr[3], arr[7], arr[11], arr[15]]
  ];
}

/**
 * 构建从相机内参的 Three.js 投影矩阵
 * 用于 Camera Matching，使 Three.js 相机与真实相机内参同步
 */
export function buildProjectionMatrix(
  K: CameraIntrinsics,
  near: number = 0.1,
  far: number = 100
): Matrix4 {
  const { fx, fy, cx, cy, width, height } = K;
  
  // OpenGL/Three.js 投影矩阵
  // 参考: https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
  const left = -cx * near / fx;
  const right = (width - cx) * near / fx;
  const bottom = -(height - cy) * near / fy;
  const top = cy * near / fy;
  
  const A = (right + left) / (right - left);
  const B = (top + bottom) / (top - bottom);
  const C = -(far + near) / (far - near);
  const D = -2 * far * near / (far - near);
  
  return [
    [2 * near / (right - left), 0, A, 0],
    [0, 2 * near / (top - bottom), B, 0],
    [0, 0, C, D],
    [0, 0, -1, 0]
  ];
}

// ============== 带约束 Umeyama 算法 ==============

/**
 * 绕 Z 轴的旋转矩阵
 */
export function rotationMatrixZ(angle: number): Matrix3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [
    [c, -s, 0],
    [s,  c, 0],
    [0,  0, 1]
  ];
}

/**
 * 从 mesh extent 和 GT bbox 计算 baseRotation
 * 基于 XY 平面的长短边对齐，绕 Z 轴旋转
 * 
 * @param meshExtent mesh 的 extent [ex, ey, ez]
 * @param gtBbox GT bbox 信息
 * @returns baseRotation (3x3)
 */
export function computeBboxBaseRotation(
  meshExtent: [number, number, number],
  gtBbox: { position: [number, number, number]; scale: [number, number, number]; R?: number[][] }
): Matrix3 {
  // 如果没有 R 矩阵，返回单位矩阵
  if (!gtBbox.R) {
    return identity3();
  }
  
  const gtR = gtBbox.R;
  const gtScale = gtBbox.scale;
  const gtExtent: [number, number, number] = [gtScale[0] * 2, gtScale[1] * 2, gtScale[2] * 2];
  
  // 过滤出主要在 XY 平面的轴（Z 分量小于 XY 分量）
  const gtEdgesXYDominant: { length: number; direction: number[]; index: number }[] = [];
  for (let i = 0; i < 3; i++) {
    const direction = [gtR[0][i], gtR[1][i], gtR[2][i]];
    const length = gtExtent[i];
    
    const zComponent = Math.abs(direction[2]);
    const xyComponent = Math.sqrt(direction[0] * direction[0] + direction[1] * direction[1]);
    
    if (xyComponent > zComponent) {
      const xyProjectionLength = length * xyComponent;
      gtEdgesXYDominant.push({ length: xyProjectionLength, direction, index: i });
    }
  }
  
  // 如果没有主要在 XY 平面的轴，返回单位矩阵
  if (gtEdgesXYDominant.length < 2) {
    return identity3();
  }
  
  // 按 XY 平面投影长度排序（从大到小）
  gtEdgesXYDominant.sort((a, b) => b.length - a.length);
  
  const gtLongestXYLen = gtEdgesXYDominant[0].length;
  const gtSecondXYLen = gtEdgesXYDominant[1].length;
  const gtXYRatio = gtLongestXYLen / (gtSecondXYLen + 1e-6);
  
  // Mesh 在 XY 平面的长短边
  const meshXLength = meshExtent[0];
  const meshYLength = meshExtent[1];
  const meshXYRatio = Math.max(meshXLength, meshYLength) / (Math.min(meshXLength, meshYLength) + 1e-6);
  
  // 只有当长短边都足够明显时才旋转（阈值：长短比 > 1.2）
  if (gtXYRatio <= 1.2 || meshXYRatio <= 1.2) {
    return identity3();
  }
  
  // GT bbox 在 XY 平面的最长边方向
  const gtLongestDirection = gtEdgesXYDominant[0].direction;
  const gtLongestXY = [gtLongestDirection[0], gtLongestDirection[1]];
  const gtLongestXYNorm = Math.sqrt(gtLongestXY[0] * gtLongestXY[0] + gtLongestXY[1] * gtLongestXY[1]);
  gtLongestXY[0] /= gtLongestXYNorm + 1e-6;
  gtLongestXY[1] /= gtLongestXYNorm + 1e-6;
  
  // Mesh 在 XY 平面的最长边方向
  const meshLongestXY = meshXLength > meshYLength ? [1, 0] : [0, 1];
  
  // 计算需要旋转的角度（绕 Z 轴）
  const gtAngle = Math.atan2(gtLongestXY[1], gtLongestXY[0]);
  const meshAngle = Math.atan2(meshLongestXY[1], meshLongestXY[0]);
  const rotationAngle = gtAngle - meshAngle;
  
  // 只有当旋转角度显著时才旋转（大于 10 度）
  if (Math.abs(rotationAngle) <= Math.PI / 18) {
    return identity3();
  }
  
  return rotationMatrixZ(rotationAngle);
}

/**
 * 计算纯 Bbox Align 变换矩阵（只使用 bbox 信息，不使用点对）
 * 
 * 核心目标：让变换后的 mesh 边与 GT bbox 边平行（使用 GT bbox 的旋转矩阵 R）
 * 
 * 步骤：
 * 1. 直接使用 GT bbox 的旋转矩阵 R（保证边边平行）
 * 2. 计算 scale（使用对角线比例）
 * 3. 计算 translation（mesh 中心对齐 GT bbox 中心）
 * 
 * @param meshInfo mesh 信息 { center, extent }
 * @param gtBbox GT bbox 信息 { position, scale, R, corners }
 * @returns 4x4 变换矩阵
 */
export function computeBboxAlignTransform(
  meshInfo: { center: [number, number, number]; extent: [number, number, number] },
  gtBbox: { position: [number, number, number]; scale: [number, number, number]; R?: number[][]; corners?: number[][] }
): Matrix4 {
  // 注意：前端 MVFrameView.tsx 中有 MESH_SCALE = 2.0 的硬编码
  const MESH_SCALE = 2.0;
  
  // 考虑 MESH_SCALE 后的 mesh 信息
  const scaledMeshCenter: [number, number, number] = [
    meshInfo.center[0] * MESH_SCALE,
    meshInfo.center[1] * MESH_SCALE,
    meshInfo.center[2] * MESH_SCALE
  ];
  const scaledMeshExtent: [number, number, number] = [
    meshInfo.extent[0] * MESH_SCALE,
    meshInfo.extent[1] * MESH_SCALE,
    meshInfo.extent[2] * MESH_SCALE
  ];
  
  // 计算 GT bbox 的 extent（优先从 corners 计算）
  let gtExtent: [number, number, number];
  let gtCenter: [number, number, number];
  
  if (gtBbox.corners && gtBbox.corners.length === 8) {
    // 从 corners 计算 extent 和 center
    const corners = gtBbox.corners;
    const min = [
      Math.min(...corners.map(c => c[0])),
      Math.min(...corners.map(c => c[1])),
      Math.min(...corners.map(c => c[2]))
    ];
    const max = [
      Math.max(...corners.map(c => c[0])),
      Math.max(...corners.map(c => c[1])),
      Math.max(...corners.map(c => c[2]))
    ];
    gtExtent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    gtCenter = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    console.log('[computeBboxAlignTransform] 使用 corners 计算 extent:', gtExtent);
  } else {
    // fallback: 使用 scale * 2（可能不准确）
    gtExtent = [gtBbox.scale[0] * 2, gtBbox.scale[1] * 2, gtBbox.scale[2] * 2];
    gtCenter = gtBbox.position;
    console.log('[computeBboxAlignTransform] 警告: 使用 scale*2 计算 extent (可能不准确)');
  }
  
  // 1. 直接使用 GT bbox 的旋转矩阵 R（保证边边平行）
  // GT bbox 的 R 矩阵定义了其局部坐标系在世界坐标系中的方向
  const gtR: Matrix3 = gtBbox.R ? [
    [gtBbox.R[0][0], gtBbox.R[0][1], gtBbox.R[0][2]],
    [gtBbox.R[1][0], gtBbox.R[1][1], gtBbox.R[1][2]],
    [gtBbox.R[2][0], gtBbox.R[2][1], gtBbox.R[2][2]]
  ] : identity3();
  
  console.log('[computeBboxAlignTransform] 使用 GT bbox R 矩阵:', gtR);
  
  // 2. 将 mesh extent 变换到 GT bbox 的局部坐标系
  // 在 GT bbox 的局部坐标系中计算各向异性 scale
  // 注意：这里假设 mesh 的 AABB 轴与世界坐标系对齐
  // 需要将 mesh extent 投影到 GT bbox 的局部坐标系
  
  // 简化处理：使用对角线比例作为各向同性 scale
  // 因为 mesh 的 AABB 轴向和 GT bbox 的轴向不一定对齐
  const meshDiagonal = Math.sqrt(
    scaledMeshExtent[0] ** 2 + scaledMeshExtent[1] ** 2 + scaledMeshExtent[2] ** 2
  );
  const gtDiagonal = Math.sqrt(gtExtent[0] ** 2 + gtExtent[1] ** 2 + gtExtent[2] ** 2);
  const scale = gtDiagonal / (meshDiagonal + 1e-6);
  
  console.log('[computeBboxAlignTransform] 使用各向同性 scale:', scale);
  
  // 3. 变换公式: P_world = scale * gtR * P_mesh + t
  // 这是各向同性缩放，保证边边平行
  
  console.log('[computeBboxAlignTransform] gtExtent:', gtExtent, 'scale:', scale);
  
  // 5. Y-up 到 Z-up 变换矩阵
  // mesh 是 Y-up 的（Three.js 默认），世界坐标系是 Z-up 的
  // R_yup_to_zup: x' = x, y' = -z, z' = y
  const R_yup_to_zup: Matrix3 = [
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
  ];
  
  // 6. 构建组合旋转矩阵: gtR * R_yup_to_zup
  const combinedR = multiply3x3(gtR, R_yup_to_zup);
  
  // 7. 重新计算 translation（使用组合旋转矩阵）
  // 让 mesh 中心对齐 GT bbox 中心: gtCenter = scale * combinedR * meshCenter + t
  // 所以: t = gtCenter - scale * combinedR * meshCenter
  const rotatedCenterCombined = multiplyMat3Vec3(combinedR, scaledMeshCenter);
  const translationCombined: Vector3 = [
    gtCenter[0] - scale * rotatedCenterCombined[0],
    gtCenter[1] - scale * rotatedCenterCombined[1],
    gtCenter[2] - scale * rotatedCenterCombined[2]
  ];
  
  console.log('[computeBboxAlignTransform] combinedR (gtR * R_yup_to_zup):', combinedR);
  console.log('[computeBboxAlignTransform] translationCombined:', translationCombined);
  
  // 8. 构建 4x4 变换矩阵: [scale * combinedR | t]
  const transformMatrix: Matrix4 = [
    [scale * combinedR[0][0], scale * combinedR[0][1], scale * combinedR[0][2], translationCombined[0]],
    [scale * combinedR[1][0], scale * combinedR[1][1], scale * combinedR[1][2], translationCombined[1]],
    [scale * combinedR[2][0], scale * combinedR[2][1], scale * combinedR[2][2], translationCombined[2]],
    [0, 0, 0, 1]
  ];
  
  return transformMatrix;
}

/**
 * 固定旋转矩阵 R，求解最优各向异性 scale 和 translation
 * 
 * 变换公式: dst = R * diag(sx, sy, sz) * src + t
 * 
 * 这里 S 是在 mesh 的局部坐标系（src 坐标系）中的缩放
 * R * S 不是正交矩阵，但这是实现各向异性缩放的正确方式
 * 
 * 求解方法：
 * 令 y = R^T * dst, x = src
 * 则 y = S * x + R^T * t = S * x + t'
 * 对每个轴分别求解 scale
 * 
 * @returns { scale, scaleVec, translation, error, transformMatrix }
 */
export function solveScaleTranslationForFixedR(
  srcPoints: Point3D[],
  dstPoints: Point3D[],
  R: Matrix3
): { scale: number; scaleVec: Vector3; translation: Vector3; error: number; transformMatrix: Matrix4 } {
  const n = srcPoints.length;
  
  if (n < 1) {
    return {
      scale: 1,
      scaleVec: [1, 1, 1],
      translation: [0, 0, 0],
      error: Infinity,
      transformMatrix: identity4()
    };
  }
  
  // 1. 将 dst 点变换到 R 的局部坐标系: y = R^T * dst
  const yPoints: Vector3[] = [];
  for (let i = 0; i < n; i++) {
    const dst: Vector3 = [dstPoints[i].x, dstPoints[i].y, dstPoints[i].z];
    // R^T * dst
    yPoints.push([
      R[0][0] * dst[0] + R[1][0] * dst[1] + R[2][0] * dst[2],
      R[0][1] * dst[0] + R[1][1] * dst[1] + R[2][1] * dst[2],
      R[0][2] * dst[0] + R[1][2] * dst[1] + R[2][2] * dst[2]
    ]);
  }
  
  // 2. src 点保持原样: x = src
  const xPoints: Vector3[] = [];
  for (let i = 0; i < n; i++) {
    xPoints.push([srcPoints[i].x, srcPoints[i].y, srcPoints[i].z]);
  }
  
  // 3. 计算质心
  const xCentroid: Vector3 = [0, 0, 0];
  const yCentroid: Vector3 = [0, 0, 0];
  
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      xCentroid[k] += xPoints[i][k];
      yCentroid[k] += yPoints[i][k];
    }
  }
  for (let k = 0; k < 3; k++) {
    xCentroid[k] /= n;
    yCentroid[k] /= n;
  }
  
  // 4. 去中心化
  const xCentered: Vector3[] = [];
  const yCentered: Vector3[] = [];
  
  for (let i = 0; i < n; i++) {
    xCentered.push([
      xPoints[i][0] - xCentroid[0],
      xPoints[i][1] - xCentroid[1],
      xPoints[i][2] - xCentroid[2]
    ]);
    yCentered.push([
      yPoints[i][0] - yCentroid[0],
      yPoints[i][1] - yCentroid[1],
      yPoints[i][2] - yCentroid[2]
    ]);
  }
  
  // 5. 对每个轴求解 scale: y[k] = s[k] * x[k]
  // 最小二乘: s[k] = Σ(y[k] * x[k]) / Σ(x[k]²)
  const numerators: Vector3 = [0, 0, 0];
  const denominators: Vector3 = [0, 0, 0];
  
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      numerators[k] += yCentered[i][k] * xCentered[i][k];
      denominators[k] += xCentered[i][k] * xCentered[i][k];
    }
  }
  
  const scaleVec: Vector3 = [
    denominators[0] > 1e-10 ? numerators[0] / denominators[0] : 1,
    denominators[1] > 1e-10 ? numerators[1] / denominators[1] : 1,
    denominators[2] > 1e-10 ? numerators[2] / denominators[2] : 1
  ];
  
  // 6. 计算 t' = yCentroid - S * xCentroid（在 R 的局部坐标系中）
  const tLocal: Vector3 = [
    yCentroid[0] - scaleVec[0] * xCentroid[0],
    yCentroid[1] - scaleVec[1] * xCentroid[1],
    yCentroid[2] - scaleVec[2] * xCentroid[2]
  ];
  
  // 7. 变换回世界坐标系: t = R * t'
  const translation: Vector3 = multiplyMat3Vec3(R, tLocal);
  
  // 8. 构建 RS = R * S 矩阵
  // RS[i][j] = R[i][j] * S[j] = R[i][j] * scaleVec[j]
  const RS: Matrix3 = [
    [R[0][0] * scaleVec[0], R[0][1] * scaleVec[1], R[0][2] * scaleVec[2]],
    [R[1][0] * scaleVec[0], R[1][1] * scaleVec[1], R[1][2] * scaleVec[2]],
    [R[2][0] * scaleVec[0], R[2][1] * scaleVec[1], R[2][2] * scaleVec[2]]
  ];
  
  // 9. 构建变换矩阵: [RS | t]
  const transformMatrix: Matrix4 = [
    [RS[0][0], RS[0][1], RS[0][2], translation[0]],
    [RS[1][0], RS[1][1], RS[1][2], translation[1]],
    [RS[2][0], RS[2][1], RS[2][2], translation[2]],
    [0, 0, 0, 1]
  ];
  
  // 10. 计算误差
  let error = 0;
  for (let i = 0; i < n; i++) {
    const transformed = transformPoint(transformMatrix, srcPoints[i]);
    const dx = transformed.x - dstPoints[i].x;
    const dy = transformed.y - dstPoints[i].y;
    const dz = transformed.z - dstPoints[i].z;
    error += dx * dx + dy * dy + dz * dz;
  }
  error = Math.sqrt(error / n);
  
  const scale = (scaleVec[0] + scaleVec[1] + scaleVec[2]) / 3;
  
  console.log(`[solveScaleTranslationForFixedR] scaleVec=[${scaleVec.map(v => v.toFixed(4)).join(', ')}], error=${error.toFixed(4)}`);
  console.log(`[solveScaleTranslationForFixedR] xCentroid=[${xCentroid.map(v => v.toFixed(3)).join(', ')}]`);
  console.log(`[solveScaleTranslationForFixedR] yCentroid=[${yCentroid.map(v => v.toFixed(3)).join(', ')}]`);
  console.log(`[solveScaleTranslationForFixedR] translation=[${translation.map(v => v.toFixed(3)).join(', ')}]`);
  
  // 调试：输出每个点的误差
  for (let i = 0; i < Math.min(n, 3); i++) {
    const transformed = transformPoint(transformMatrix, srcPoints[i]);
    console.log(`  Point ${i}: src=(${srcPoints[i].x.toFixed(3)}, ${srcPoints[i].y.toFixed(3)}, ${srcPoints[i].z.toFixed(3)}) -> (${transformed.x.toFixed(3)}, ${transformed.y.toFixed(3)}, ${transformed.z.toFixed(3)}), dst=(${dstPoints[i].x.toFixed(3)}, ${dstPoints[i].y.toFixed(3)}, ${dstPoints[i].z.toFixed(3)})`);
  }
  
  return { scale, scaleVec, translation, error, transformMatrix };
}

/**
 * 带约束的 Umeyama 算法
 * 测试 4 个角度 [0°, 90°, 180°, 270°]，固定 R 求解 s, t
 * 
 * @param srcPoints 源点集 (模型空间局部坐标)
 * @param dstPoints 目标点集 (世界空间坐标)
 * @param baseRotation 基础旋转矩阵（从 Bbox Align 计算）
 * @returns 最佳对齐结果
 */
export function solveUmeyamaConstrained(
  srcPoints: Point3D[],
  dstPoints: Point3D[],
  baseRotation: Matrix3
): UmeyamaResult {
  const n = srcPoints.length;
  
  if (n < 3) {
    console.warn('Constrained Umeyama requires at least 3 point pairs');
    return {
      rotation: identity3(),
      translation: [0, 0, 0],
      scale: 1,
      transformMatrix: identity4(),
      error: Infinity
    };
  }
  
  // 测试 4 个角度
  const angles = [0, Math.PI / 2, Math.PI, 3 * Math.PI / 2];
  let bestResult: { scale: number; scaleVec: Vector3; translation: Vector3; error: number; transformMatrix: Matrix4 } | null = null;
  let bestRotation: Matrix3 = identity3();
  let bestError = Infinity;
  
  console.log('[Constrained Umeyama] baseRotation:', baseRotation);
  
  for (const angle of angles) {
    // R_test = baseRotation * rotZ（先绕 mesh 的 Z 轴旋转，再应用 GT bbox 的 R）
    const rotZ = rotationMatrixZ(angle);
    const R_test = multiply3x3(baseRotation, rotZ);
    
    // 固定 R_test，求解 s, t
    const result = solveScaleTranslationForFixedR(srcPoints, dstPoints, R_test);
    
    console.log(`[Constrained Umeyama] angle=${(angle * 180 / Math.PI).toFixed(0)}°, scaleVec=[${result.scaleVec.map(v => v.toFixed(4)).join(', ')}], error=${result.error.toFixed(4)}`);
    
    if (result.error < bestError) {
      bestError = result.error;
      bestResult = result;
      bestRotation = R_test;
    }
  }
  
  if (!bestResult) {
    return solveUmeyama(srcPoints, dstPoints);
  }
  
  console.log('[Constrained Umeyama] best angle error:', bestError.toFixed(4));
  
  return {
    rotation: bestRotation,
    translation: bestResult.translation,
    scale: bestResult.scale,
    transformMatrix: bestResult.transformMatrix,
    error: bestResult.error
  };
}

/**
 * RANSAC + 带约束 Umeyama 算法
 */
export function solveUmeyamaConstrainedRANSAC(
  srcPoints: Point3D[],
  dstPoints: Point3D[],
  baseRotation: Matrix3,
  options: {
    maxIterations?: number;
    inlierThreshold?: number;
  } = {}
): UmeyamaResult & { inlierIndices: number[]; outlierIndices: number[] } {
  const n = srcPoints.length;
  const maxIterations = options.maxIterations ?? 100;
  const inlierThreshold = options.inlierThreshold ?? 0.05;
  
  if (n < 3) {
    return {
      ...solveUmeyamaConstrained(srcPoints, dstPoints, baseRotation),
      inlierIndices: [],
      outlierIndices: Array.from({ length: n }, (_, i) => i)
    };
  }
  
  if (n <= 4) {
    const result = solveUmeyamaConstrained(srcPoints, dstPoints, baseRotation);
    return {
      ...result,
      inlierIndices: Array.from({ length: n }, (_, i) => i),
      outlierIndices: []
    };
  }
  
  let bestResult: UmeyamaResult | null = null;
  let bestInliers: number[] = [];
  let bestError = Infinity;
  
  for (let iter = 0; iter < maxIterations; iter++) {
    // 随机选择 3 个点
    const sampleIndices = randomSample(n, 3);
    const sampleSrc = sampleIndices.map(i => srcPoints[i]);
    const sampleDst = sampleIndices.map(i => dstPoints[i]);
    
    // 用样本点计算带约束变换
    const candidateResult = solveUmeyamaConstrained(sampleSrc, sampleDst, baseRotation);
    
    // 计算所有点的误差，找内点
    const inliers: number[] = [];
    let totalError = 0;
    
    for (let i = 0; i < n; i++) {
      const transformed = transformPoint(candidateResult.transformMatrix, srcPoints[i]);
      const dx = transformed.x - dstPoints[i].x;
      const dy = transformed.y - dstPoints[i].y;
      const dz = transformed.z - dstPoints[i].z;
      const error = Math.sqrt(dx * dx + dy * dy + dz * dz);
      
      if (error < inlierThreshold) {
        inliers.push(i);
        totalError += error;
      }
    }
    
    if (inliers.length > bestInliers.length ||
        (inliers.length === bestInliers.length && totalError < bestError)) {
      bestInliers = inliers;
      bestError = totalError;
      bestResult = candidateResult;
    }
    
    if (inliers.length / n >= 0.9) {
      break;
    }
  }
  
  // 用所有内点重新计算
  if (bestInliers.length >= 3) {
    const inlierSrc = bestInliers.map(i => srcPoints[i]);
    const inlierDst = bestInliers.map(i => dstPoints[i]);
    bestResult = solveUmeyamaConstrained(inlierSrc, inlierDst, baseRotation);
  }
  
  const outlierIndices = Array.from({ length: n }, (_, i) => i)
    .filter(i => !bestInliers.includes(i));
  
  console.log(`[Constrained RANSAC] 内点: ${bestInliers.length}/${n}, 离群点: ${outlierIndices.length}`);
  
  return {
    ...(bestResult ?? solveUmeyamaConstrained(srcPoints, dstPoints, baseRotation)),
    inlierIndices: bestInliers,
    outlierIndices
  };
}
