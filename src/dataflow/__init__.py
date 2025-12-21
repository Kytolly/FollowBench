from configs import CONFIG
import logging
logger = logging.getLogger(__name__)

REQUIRED_META_KEYS = {'team_name', 'model_name', "modal", "mode", "contact"}
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov'}

try:
    SUB_CFG = CONFIG['submission']
    TOTAL_CASES_NUM = SUB_CFG['total_cases_num']
    STANDARD_RESOLUTION = (SUB_CFG['resolution_height'], SUB_CFG['resolution_width']) # (H, W)
    STANDARD_CLIP_LEN = SUB_CFG['clip_len']
    STANDARD_CLIP_FPS = SUB_CFG['clip_fps']
except KeyError:
    logger.error("Missing submission config in CONFIG. Using defaults/empty.")
    TOTAL_CASES_NUM = 0
    STANDARD_RESOLUTION = (0, 0)
    STANDARD_CLIP_LEN = 0
    STANDARD_CLIP_FPS = 0