#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль для проверки фото с использованием Chain of Responsibility + Strategy паттернов.
Масштабируемая архитектура с разделением ответственности.
"""

import math
import os
import sys
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

# ============== Pydantic Models ==============


class ImageSize(BaseModel):
    width_px: int
    height_px: int


class Metadata(BaseModel):
    image_size: Optional[ImageSize] = None
    background_white_ratio_percent: Optional[float] = None
    faces_count: Optional[int] = None
    centering_horizontal_percent: Optional[float] = None
    centering_vertical_percent: Optional[float] = None
    eye_angle_degrees: Optional[float] = None
    pitch_angle_degrees: Optional[float] = None


class Result(BaseModel):
    success: bool
    code: Optional[str] = None
    text: Optional[str] = None
    metadata: Metadata
    debug_image_path: Optional[str] = None


class FaceData(BaseModel):
    """Данные обнаруженного лица"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: tuple[float, float, float, float]  # x, y, w, h
    landmarks: np.ndarray  # 5 landmarks


class PhotoContext(BaseModel):
    """Контекст для передачи между валидаторами"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Optional[np.ndarray] = None
    image_size: Optional[ImageSize] = None
    face_data: Optional[FaceData] = None
    metadata: Metadata = Metadata()


# ============== Face Detector (Strategy) ==============


class IFaceDetector(ABC):
    """Интерфейс для детекторов лиц"""

    @abstractmethod
    def detect(self, image: np.ndarray, image_size: ImageSize) -> Optional[FaceData]:
        pass


class YuNetFaceDetector(IFaceDetector):
    """Детектор лиц на основе YuNet"""

    def __init__(self, model_path: str = "face_detection_yunet_2023mar.onnx"):
        self.model_path = model_path
        self._ensure_model()

    def _ensure_model(self):
        if not os.path.exists(self.model_path):
            print("Скачивание модели YuNet...")
            urllib.request.urlretrieve(
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                self.model_path,
            )

    def detect(self, image: np.ndarray, image_size: ImageSize) -> Optional[FaceData]:
        detector = cv2.FaceDetectorYN.create(
            self.model_path, "", (image_size.width_px, image_size.height_px)
        )
        faces = detector.detect(image)[1]

        if faces is None or len(faces) == 0:
            return None

        if len(faces) > 1:
            raise ValueError("Multiple faces detected")

        face = faces[0]
        bbox = tuple(face[:4])
        landmarks = face[4:14].reshape(5, 2)

        return FaceData(bbox=bbox, landmarks=landmarks)


# ============== Validators (Chain of Responsibility) ==============


class IValidator(ABC):
    """Базовый интерфейс валидатора"""

    @abstractmethod
    def validate(self, context: PhotoContext) -> Optional[Result]:
        """Возвращает Result при ошибке, None при успехе"""
        pass


class BackgroundValidator(IValidator):
    """Валидатор белого фона"""

    def __init__(self, min_white_ratio: float = 0.3):
        self.min_white_ratio = min_white_ratio

    def validate(self, context: PhotoContext) -> Optional[Result]:
        if context.image is None:
            return None

        hsv = cv2.cvtColor(context.image, cv2.COLOR_BGR2HSV)
        white_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 200)
        white_pixels = cv2.countNonZero(white_mask.astype(np.uint8))
        total_pixels = context.image_size.width_px * context.image_size.height_px
        white_ratio = white_pixels / total_pixels

        context.metadata.background_white_ratio_percent = round(white_ratio * 100, 1)

        if white_ratio < self.min_white_ratio:
            return Result(
                success=False,
                code="background_not_white",
                text="Фон не белый",
                metadata=context.metadata,
            )
        return None


class FaceDetectionValidator(IValidator):
    """Валидатор обнаружения лица"""

    def __init__(self, detector: IFaceDetector):
        self.detector = detector

    def validate(self, context: PhotoContext) -> Optional[Result]:
        try:
            face_data = self.detector.detect(context.image, context.image_size)
        except ValueError as e:
            context.metadata.faces_count = None
            return Result(
                success=False,
                code="multiple_faces",
                text="Найдено несколько лиц",
                metadata=context.metadata,
            )

        if face_data is None:
            context.metadata.faces_count = 0
            return Result(
                success=False,
                code="face_not_recognized",
                text="Лицо не распознано",
                metadata=context.metadata,
            )

        context.metadata.faces_count = 1
        context.face_data = face_data
        return None


class CenteringValidator(IValidator):
    """Валидатор центрирования лица"""

    def __init__(self, horizontal_threshold: float = 0.1, vertical_threshold: float = 0.25):
        self.horizontal_threshold = horizontal_threshold
        self.vertical_threshold = vertical_threshold

    def validate(self, context: PhotoContext) -> Optional[Result]:
        if context.face_data is None:
            return None

        nose = context.face_data.landmarks[2]
        center_x = context.image_size.width_px / 2
        center_y = context.image_size.height_px / 2

        offset_x = abs(nose[0] - center_x) / center_x
        context.metadata.centering_horizontal_percent = round(offset_x * 100, 1)

        offset_y = abs(nose[1] - center_y) / center_y
        context.metadata.centering_vertical_percent = round(offset_y * 100, 1)

        if nose[0] > center_x + self.horizontal_threshold * center_x:
            return Result(
                success=False,
                code="centering_right",
                text="Лицо смещено вправо",
                metadata=context.metadata,
            )
        if nose[0] < center_x - self.horizontal_threshold * center_x:
            return Result(
                success=False,
                code="centering_left",
                text="Лицо смещено влево",
                metadata=context.metadata,
            )
        if nose[1] < center_y - self.vertical_threshold * center_y:
            return Result(
                success=False,
                code="centering_up",
                text="Лицо смещено вверх",
                metadata=context.metadata,
            )
        if nose[1] > center_y + self.vertical_threshold * center_y:
            return Result(
                success=False,
                code="centering_down",
                text="Лицо смещено вниз",
                metadata=context.metadata,
            )
        return None


class TiltValidator(IValidator):
    """Валидатор наклона головы"""

    def __init__(self, max_angle: float = 15.0):
        self.max_angle = max_angle

    def validate(self, context: PhotoContext) -> Optional[Result]:
        if context.face_data is None:
            return None

        landmarks = context.face_data.landmarks
        left_eye = landmarks[1]
        right_eye = landmarks[0]
        nose = landmarks[2]

        # Roll angle
        delta_y = left_eye[1] - right_eye[1]
        delta_x = left_eye[0] - right_eye[0]
        eye_angle = math.degrees(math.atan2(delta_y, delta_x))
        context.metadata.eye_angle_degrees = round(eye_angle, 2)

        # Pitch angle
        eye_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )
        dx = nose[0] - eye_center[0]
        dy = nose[1] - eye_center[1]
        pitch_angle = math.degrees(math.atan2(dy, dx)) - 90
        context.metadata.pitch_angle_degrees = round(pitch_angle, 2)

        if eye_angle > self.max_angle:
            return Result(
                success=False,
                code="tilt_left",
                text="Голова наклонена влево",
                metadata=context.metadata,
            )
        if eye_angle < -self.max_angle:
            return Result(
                success=False,
                code="tilt_right",
                text="Голова наклонена вправо",
                metadata=context.metadata,
            )
        if pitch_angle > self.max_angle:
            return Result(
                success=False,
                code="tilt_down",
                text="Голова наклонена вниз",
                metadata=context.metadata,
            )
        if pitch_angle < -self.max_angle:
            return Result(
                success=False,
                code="tilt_up",
                text="Голова наклонена вверх",
                metadata=context.metadata,
            )
        return None


# ============== Debug Visualizer ==============


class DebugVisualizer:
    """Класс для визуализации ошибок на border_black.png"""

    def __init__(self, border_path: str = "faces/img/border_black.png"):
        self.border_path = border_path

    def create_debug_image(
        self, context: PhotoContext, output_path: str = "debug_output.jpg"
    ) -> str:
        """Создаёт изображение с наложением фото на border_black.png"""
        if context.image is None:
            return ""

        # Загружаем border_black.png
        if not os.path.exists(self.border_path):
            # Если border_black.png не существует, создаём чёрную рамку
            target_size = (1000, 1000)
            border_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        else:
            border_image = cv2.imread(self.border_path)

        border_h, border_w = border_image.shape[:2]
        img_h, img_w = context.image.shape[:2]

        # Масштабируем изображение под размер border
        kx = border_w / img_w
        ky = border_h / img_h
        final_image = cv2.resize(context.image, (border_w, border_h))
        offset_x = 0
        offset_y = 0

        # Конвертируем в PIL для работы с прозрачностью
        border_pil = Image.fromarray(cv2.cvtColor(border_image, cv2.COLOR_BGR2RGB)).convert("RGBA")
        photo_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)).convert("RGBA")

        # Делаем фото полупрозрачным (альфа 128 = 50%)
        photo_data = photo_pil.getdata()
        new_photo_data = [(r, g, b, 128) for r, g, b, a in photo_data]
        photo_pil.putdata(new_photo_data)

        # Накладываем полупрозрачное фото на border
        result_image = Image.alpha_composite(border_pil, photo_pil)

        # Рисуем landmarks если есть
        draw = ImageDraw.Draw(result_image)

        if context.face_data and context.face_data.landmarks is not None:
            landmarks = context.face_data.landmarks
            # Рисуем точки landmarks с учётом смещения
            for point in landmarks:
                x = (point[0] - offset_x) * kx
                y = (point[1] - offset_y) * ky
                radius = 8
                draw.ellipse(
                    [(x - radius, y - radius), (x + radius, y + radius)],
                    fill=(255, 0, 0, 255),
                )

            # Рисуем линии между глазами
            left_eye = landmarks[1]
            right_eye = landmarks[0]
            draw.line(
                [
                    ((right_eye[0] - offset_x) * kx, (right_eye[1] - offset_y) * ky),
                    ((left_eye[0] - offset_x) * kx, (left_eye[1] - offset_y) * ky),
                ],
                fill=(0, 255, 0, 255),
                width=3,
            )

            # Рисуем нос (центральная точка)
            nose = landmarks[2]
            draw.ellipse(
                [
                    ((nose[0] - offset_x) * kx - 10, (nose[1] - offset_y) * ky - 10),
                    ((nose[0] - offset_x) * kx + 10, (nose[1] - offset_y) * ky + 10),
                ],
                fill=(0, 255, 255, 255),
            )

        # Рисуем центр изображения
        center_x, center_y = border_w / 2, border_h / 2
        draw.line(
            [(center_x - 20, center_y), (center_x + 20, center_y)],
            fill=(255, 255, 0, 255),
            width=2,
        )
        draw.line(
            [(center_x, center_y - 20), (center_x, center_y + 20)],
            fill=(255, 255, 0, 255),
            width=2,
        )

        # Сохраняем результат (конвертируем в RGB для JPEG)
        result_image = result_image.convert("RGB")
        result_array = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_array)
        return output_path


# ============== Pipeline (Chain of Responsibility Manager) ==============


class ValidationPipeline:
    """Менеджер цепочки валидаторов"""

    def __init__(self, validators: List[IValidator]):
        self.validators = validators

    def execute(self, context: PhotoContext) -> Result:
        """Выполняет все валидаторы по порядку"""
        for validator in self.validators:
            result = validator.validate(context)
            if result:
                return result

        return Result(success=True, metadata=context.metadata)


# ============== Photo Analyzer (Facade) ==============


class PhotoAnalyzer:
    """Фасад для анализа фото"""

    def __init__(
        self,
        image_path: str,
        detector: Optional[IFaceDetector] = None,
        validators: Optional[List[IValidator]] = None,
    ):
        self.image_path = image_path
        self.detector = detector or YuNetFaceDetector()
        self.visualizer = DebugVisualizer()

        if validators is None:
            validators = [
                BackgroundValidator(),
                FaceDetectionValidator(self.detector),
                CenteringValidator(),
                TiltValidator(),
            ]

        self.pipeline = ValidationPipeline(validators)

    def analyze(self) -> Result:
        """Анализирует фото и возвращает результат"""
        context = PhotoContext()

        # Загрузка изображения
        image = cv2.imread(self.image_path)
        if image is None:
            return Result(
                success=False,
                code="load_error",
                text="Ошибка загрузки файла",
                metadata=context.metadata,
            )

        h, w = image.shape[:2]
        context.image = image
        context.image_size = ImageSize(width_px=w, height_px=h)
        context.metadata.image_size = context.image_size

        # Выполнение валидации
        result = self.pipeline.execute(context)

        # Создание debug изображения при ошибке
        if not result.success:
            # Создаём директорию outputs если не существует
            os.makedirs("outputs", exist_ok=True)

            # Формируем имя файла: cn_<исходное_имя>
            base_name = os.path.basename(self.image_path)
            name_without_ext = os.path.splitext(base_name)[0]
            ext = os.path.splitext(base_name)[1]
            debug_filename = f"cn_{name_without_ext}{ext}"
            debug_path = os.path.join("outputs", debug_filename)

            debug_path = self.visualizer.create_debug_image(context, debug_path)
            result.debug_image_path = debug_path

        return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python photo.py <путь_к_фото>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден")
        sys.exit(1)

    analyzer = PhotoAnalyzer(image_path)
    result = analyzer.analyze()
    print(result.model_dump_json(indent=2, exclude_unset=True))
