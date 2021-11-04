import cv2
import CalibrationHelpers as calib
intrinsics, distortion, roi, new_intrinsics = calib.CalibrateCamera('calibration_data',True)
cv2.destroyAllWindows()  
calib.SaveCalibrationData('calibration_data', intrinsics, distortion, new_intrinsics, roi)