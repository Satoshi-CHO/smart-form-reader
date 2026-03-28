import cv2
import numpy as np
import os
import json

# =====================================================================
# 1. 画像処理の各ステップを個別のクラスへ分離 (単一責任の原則: SRP)
# これにより各クラスの働きが明確化し、テストや再利用が容易になります。
# =====================================================================

class ImageMatcher:
    """
    テンプレート画像と入力画像の特徴量マッチング（AKAZE）のみを担当するクラス
    """
    def __init__(self, template_img: np.ndarray):
        self.template_img = template_img
        self.akaze = cv2.AKAZE_create()
        # テンプレートの特徴量をあらかじめ計算しておく（パフォーマンス向上）
        self.kp_t, self.des_t = self.akaze.detectAndCompute(self.template_img, None)

    def match(self, target_img: np.ndarray):
        """対象画像とのマッチングを行い、(キーポイント, 良好なマッチングリスト)を返す"""
        if self.des_t is None:
            return None, []
            
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) if len(target_img.shape) == 3 else target_img
        kp_target, des_target = self.akaze.detectAndCompute(target_gray, None)
        
        if des_target is None:
            return kp_target, []

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(self.des_t, des_target, k=2)
        
        # Lowe's ratio testを用いた良いマッチングだけをフィルター
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance] if len(matches) > 0 and len(matches[0]) == 2 else []
        
        return kp_target, good_matches


class ImageAligner:
    """
    マッチング結果を基に、スケール補正、回転補正、ホモグラフィ変換を行うクラス
    """
    @staticmethod
    def align(template_img: np.ndarray, template_kps: list, target_img: np.ndarray, target_kps: list, good_matches: list) -> np.ndarray:
        """Homography変換を用いて画像をテンプレートに合わせて変形する"""
        if len(good_matches) < 10:
            return None
            
        src_pts = np.float32([template_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([target_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSACを用いて外れ値を除少しながら変換行列を計算
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return None

        h, w = template_img.shape[:2]
        aligned_img = cv2.warpPerspective(target_img, M, (w, h))
        return aligned_img


class RoiExtractor:
    """
    指定された座標設定（マスク）に基づいて画像を切り出す処理のみを担当するクラス
    """
    @staticmethod
    def extract_and_save(aligned_img: np.ndarray, masks_config: list, output_dir: str, file_id: str) -> list:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = []
        for index, mask in enumerate(masks_config):
            roi_id = mask.get("id", f"roi_{index}")
            x, y, w, h = mask.get("x", 0), mask.get("y", 0), mask.get("w", 0), mask.get("h", 0)

            # 画像範囲外のクロップを防ぐための安全処理
            ih, iw = aligned_img.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(iw, x + w), min(ih, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            cropped = aligned_img[y1:y2, x1:x2]
            out_filename = f"{file_id}_{roi_id}.png"
            out_path = os.path.join(output_dir, out_filename)
            
            # 画像の保存作業もこのクラスの責任範囲とします
            cv2.imwrite(out_path, cropped)
            
            results.append({
                "roi_id": roi_id,
                "crop_path": out_path,
                "type": mask.get("type", "text"),
                "expected_length": mask.get("expected_length", -1)
            })
            
        return results


# =====================================================================
# 2. アプリケーション層 (Facade パターン)
# 上記で細分化されたクラス群をオーケストレーション(連携)させる役割です。
# main_gui.py など外部からはこのクラスだけを使えば良いように設計（Facade）します。
# =====================================================================

class ImageProcessor:
    def __init__(self, template_path, masks_config_path):
        self.template_path = template_path
        self.masks_config_path = masks_config_path
        
        # テンプレート画像の読み込み
        if os.path.exists(template_path):
            self.template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            # Matcherクラスの初期化
            self.matcher = ImageMatcher(self.template_img)
        else:
            self.template_img = None
            self.matcher = None
            
        # 設定の読み込み
        if os.path.exists(masks_config_path):
            with open(masks_config_path, 'r', encoding='utf-8') as f:
                self.masks_config = json.load(f)
        else:
            self.masks_config = []

    def align_and_correct_rotation(self, image_path):
        """
        画像の読み込み・回転検知・位置合わせの一連のフローを制御（Facade）します。
        具体的な処理内容（アルゴリズム等）はmatcherやalignerなどの別クラスに任せます。
        """
        img = cv2.imread(image_path)
        if img is None or self.template_img is None or self.matcher is None:
            return None, "IMAGE_LOAD_ERROR"

        # 180度回転した画像も用意
        img_rotated = cv2.rotate(img, cv2.ROTATE_180)

        # Matcherを用いて特徴点とマッチング結果を取得（0度 および 180度）
        kp_0, good_0 = self.matcher.match(img)
        kp_180, good_180 = self.matcher.match(img_rotated)

        if len(good_0) < 10 and len(good_180) < 10:
            return None, "CROP_FAILED"

        # 最もマッチした方向（正しい向き）を採用するロジック
        if len(good_180) > len(good_0) * 1.5:
            best_img, best_kp, best_good = img_rotated, kp_180, good_180
        else:
            best_img, best_kp, best_good = img, kp_0, good_0

        # Alignerを用いてホモグラフィ変換・位置合わせを実施
        aligned_img = ImageAligner.align(self.template_img, self.matcher.kp_t, best_img, best_kp, best_good)
        
        if aligned_img is None:
            return None, "CROP_FAILED"

        return aligned_img, "SUCCESS"

    def extract_rois(self, aligned_img, output_dir, file_id):
        """
        位置合わせされた画像からROIを切り出します。
        実際の処理内容は RoiExtractor に委譲しています。
        """
        return RoiExtractor.extract_and_save(aligned_img, self.masks_config, output_dir, file_id)
