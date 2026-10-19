"""集中管理路徑與憑證來源。所有機密只從環境變數/.env 取得，不寫死在程式碼裡。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# override=False：已存在的環境變數優先，方便 CI 或臨時覆寫
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _root(env_key: str, default_name: str) -> Path:
    value = os.getenv(env_key, "").strip()
    path = Path(value).expanduser() if value else PROJECT_ROOT / default_name
    return path.resolve()


DATA_ROOT = _root("DATA_ROOT", "dataset")
RESULT_ROOT = _root("RESULT_ROOT", "results")

RAW_DIR = DATA_ROOT / "raw"          # 原始下載，盡量保持與來源一致
PROCESSED_DIR = DATA_ROOT / "processed"  # 清洗、調整、對齊後的資料
META_DIR = DATA_ROOT / "meta"        # 股票清單、產業別、下載紀錄


@dataclass(frozen=True)
class FubonCredentials:
    """富邦 Neo SDK 登入所需資訊。"""

    personal_id: str
    password: str
    cert_path: str
    cert_password: str

    @classmethod
    def from_env(cls) -> "FubonCredentials":
        missing = [
            key
            for key in ("FUBON_ID", "FUBON_PASSWORD", "FUBON_CERT_PATH")
            if not os.getenv(key)
        ]
        if missing:
            raise RuntimeError(
                f"缺少環境變數 {', '.join(missing)}。請複製 .env.example 成 .env 並填入。"
            )
        cert_path = Path(os.environ["FUBON_CERT_PATH"]).expanduser()
        if not cert_path.exists():
            raise FileNotFoundError(f"找不到憑證檔：{cert_path}")
        return cls(
            personal_id=os.environ["FUBON_ID"],
            password=os.environ["FUBON_PASSWORD"],
            cert_path=str(cert_path),
            # 部分憑證無密碼，允許空字串
            cert_password=os.getenv("FUBON_CERT_PASSWORD", ""),
        )

    def __repr__(self) -> str:  # 避免不小心把密碼印進 log
        return f"FubonCredentials(personal_id={self.personal_id[:4]}***, cert={Path(self.cert_path).name})"


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, META_DIR, RESULT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
