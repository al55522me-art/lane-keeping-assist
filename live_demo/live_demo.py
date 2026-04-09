import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import time

class LaneKeepingAssist:
    def __init__(self, model_path='tusimple_best.pth'):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Модель
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                             in_channels=3, classes=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.input_size = (800, 288)

    def detect_lanes(self, frame):
        """Детекция полос"""
        orig_h, orig_w = frame.shape[:2]

        # Подготовка
        frame_resized = cv2.resize(frame, self.input_size)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        frame_tensor = frame_tensor / 255.0
        frame_tensor = (frame_tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        with torch.no_grad():
            pred = torch.sigmoid(self.model(frame_tensor))
            pred_mask = (pred > 0.3).squeeze().cpu().numpy()

        pred_mask_orig = cv2.resize(pred_mask.astype('uint8') * 255, (orig_w, orig_h))

        return pred_mask_orig

    def calculate_steering(self, mask):
        """Вычислить команду поворота"""
        moments = cv2.moments(mask)
        if moments["m00"] > 1000:  # Есть полосы
            cx = int(moments["m10"] / moments["m00"])
            car_center = mask.shape[1] // 2

            offset = cx - car_center
            offset_norm = offset / car_center

            if abs(offset_norm) < 0.05:
                return "ПРЯМО", offset_norm
            elif offset_norm < 0:
                return "ВПРАВО", offset_norm
            else:
                return "ВЛЕВО", offset_norm
        return "ПОЛОСЫ НЕ ВИДНЫ", 0.0

    def run_live(self):
        """Реал-тайм с вебкамеры"""
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция
            mask = self.detect_lanes(frame)

            # Команда
            command, offset = self.calculate_steering(mask)

            # Визуализация
            result = frame.copy()
            result[mask > 127] = [0, 0, 255]  # Красные полосы

            # Информация
            cv2.putText(result, command, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result, f"Offset: {offset:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Lane Keeping Assist', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    lka = LaneKeepingAssist()
    lka.run_live()