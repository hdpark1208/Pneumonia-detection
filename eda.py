from pathlib import Path
import pandas as pd


BASE_DIR = Path(r"C:\Users\PHD\.cache\kagglehub\datasets\iamtapendu\rsna-pneumonia-processed-dataset\versions\1")

TRAIN_DIR = BASE_DIR / "Training"
TEST_DIR = BASE_DIR / "Test"

TRAIN_META = BASE_DIR / "stage2_train_metadata.csv"
TEST_META = BASE_DIR / "stage2_test_metadata.csv"


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def list_directory(path: Path, max_depth: int = 2, prefix: str = ""):
    if not path.exists():
        print(f"[경고] 경로가 존재하지 않습니다: {path}")
        return

    print(f"{prefix}{path.name}/")
    if max_depth == 0:
        return

    try:
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except PermissionError:
        print(f"{prefix}  [권한 없음]")
        return

    for item in items[:30]:  # 너무 많이 나오지 않게 제한
        if item.is_dir():
            list_directory(item, max_depth=max_depth - 1, prefix=prefix + "  ")
        else:
            print(f"{prefix}  {item.name}")

    if len(items) > 30:
        print(f"{prefix}  ... ({len(items) - 30}개 더 있음)")


def inspect_csv(csv_path: Path, name: str):
    print_section(f"{name} CSV 확인")

    if not csv_path.exists():
        print(f"[오류] 파일이 없습니다: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    print(f"파일 경로: {csv_path}")
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns.tolist()}")

    print("\n[상위 5개 행]")
    print(df.head())

    print("\n[기본 정보]")
    print(df.info())

    print("\n[결측치 개수]")
    print(df.isnull().sum())

    print("\n[수치형 컬럼 요약]")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        print(df[numeric_cols].describe())
    else:
        print("수치형 컬럼 없음")

    return df


def inspect_image_files(folder: Path, name: str):
    print_section(f"{name} 폴더 이미지 파일 확인")

    if not folder.exists():
        print(f"[오류] 폴더가 없습니다: {folder}")
        return

    all_files = [p for p in folder.rglob("*") if p.is_file()]
    print(f"전체 파일 수: {len(all_files)}")

    suffix_count = {}
    for file_path in all_files:
        suffix = file_path.suffix.lower()
        suffix_count[suffix] = suffix_count.get(suffix, 0) + 1

    print("확장자별 개수:")
    for k, v in sorted(suffix_count.items()):
        print(f"  {k}: {v}")

    print("\n샘플 파일 10개:")
    for p in all_files[:10]:
        print(p.relative_to(folder))


def main():
    print_section("기본 경로 확인")
    print(f"BASE_DIR   : {BASE_DIR}")
    print(f"TRAIN_DIR  : {TRAIN_DIR}  | exists={TRAIN_DIR.exists()}")
    print(f"TEST_DIR   : {TEST_DIR}   | exists={TEST_DIR.exists()}")
    print(f"TRAIN_META : {TRAIN_META} | exists={TRAIN_META.exists()}")
    print(f"TEST_META  : {TEST_META}  | exists={TEST_META.exists()}")

    print_section("폴더 구조 미리보기")
    list_directory(BASE_DIR, max_depth=2)

    train_df = inspect_csv(TRAIN_META, "TRAIN META")
    test_df = inspect_csv(TEST_META, "TEST META")

    inspect_image_files(TRAIN_DIR, "TRAINING")
    inspect_image_files(TEST_DIR, "TEST")

    print_section("추가 체크")
    if train_df is not None:
        for col in train_df.columns:
            unique_count = train_df[col].nunique(dropna=False)
            print(f"{col}: unique={unique_count}")

    if test_df is not None:
        for col in test_df.columns:
            unique_count = test_df[col].nunique(dropna=False)
            print(f"{col}: unique={unique_count}")

#%%
# import pandas as pd
# from pathlib import Path

# csv_path = Path(r"C:\Users\PHD\.cache\kagglehub\datasets\iamtapendu\rsna-pneumonia-processed-dataset\versions\1\stage2_train_metadata.csv")
# df = pd.read_csv(csv_path)

# print("전체 행 수:", len(df))
# print("고유 patientId 수:", df["patientId"].nunique())
# print("\nTarget 분포:")
# print(df["Target"].value_counts())

# print("\nclass 분포:")
# print(df["class"].value_counts())

# print("\n같은 patientId가 여러 번 나온 예시 10개:")
# dup_counts = df["patientId"].value_counts()
# print(dup_counts[dup_counts > 1].head(10))

#%%
# from pathlib import Path

# base = Path(r"C:\Users\PHD\.cache\kagglehub\datasets\iamtapendu\rsna-pneumonia-processed-dataset\versions\1\Training")

# for folder_name in ["Images", "Masks"]:
#     folder = base / folder_name
#     files = [p for p in folder.rglob("*") if p.is_file()]
#     print(f"\n[{folder_name}]")
#     print("파일 수:", len(files))
#     print("샘플 10개:")
#     for p in files[:10]:
#         print(p.name)
#%%
# from pathlib import Path
# import pandas as pd
# BASE_DIR = Path(r"C:\Users\PHD\.cache\kagglehub\datasets\iamtapendu\rsna-pneumonia-processed-dataset\versions\1")
# TEST_META = BASE_DIR / "stage2_test_metadata.csv"

# df = pd.read_csv(TEST_META)
# print(df.shape)
# print(df.columns.tolist())
# print(df.head())

#%%
if __name__ == "__main__":
    main()