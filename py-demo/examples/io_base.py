import os
import shutil
import zipfile

tmp_dir = "/tmp/test"

# example: copy, move and delete


def test_io_copy():
    # copy file
    shutil.copy(f"{tmp_dir}/input.txt", f"{tmp_dir}/input_cp1.txt")
    # 保留文件的元数据 (如创建时间, 修改时间)
    shutil.copy2(f"{tmp_dir}/input.txt", f"{tmp_dir}/output_cp2.txt")

    # copy dir
    shutil.copytree(f"{tmp_dir}/bak", f"{tmp_dir}/bak2", dirs_exist_ok=True)


def test_io_move():
    # move file
    shutil.move(f"{tmp_dir}/input.txt", f"{tmp_dir}/input_mv1.txt")

    # move dir
    shutil.move(f"{tmp_dir}/bak", f"{tmp_dir}/bak2")


def test_io_delete():
    # delete file
    os.remove(f"{tmp_dir}/input.txt")

    # print file to be delete
    dst_dir = f"{tmp_dir}/bak"
    for root, dirs, files in os.walk(dst_dir):
        for d in dirs:
            print(f"to delete dir: {os.path.join(root, d)}")
        for f in files:
            print(f"to delete file: {os.path.join(root, f)}")

    # delete dir
    # ignore_errors 忽略权限错误
    shutil.rmtree(dst_dir, ignore_errors=True)


# example: zip and upzip


def test_create_zip():
    # create zip
    # compression: ZIP_STORED (不压缩, 快), ZIP_DEFLATED (标准压缩), ZIP_BZIP2, ZIP_LZMA
    # compresslevel: 范围 0-9, 其中 9 的压缩程度最高, 但速度也最慢. 默认值 6
    with zipfile.ZipFile(
        f"{tmp_dir}/bak.zip",
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        zf.write("data.csv")
        zf.write("report.pdf")

    # append file to zip
    with zipfile.ZipFile(f"{tmp_dir}/bak.zip", mode="a") as zf:
        zf.write("log.txt")


def test_read_zip():
    with zipfile.ZipFile(f"{tmp_dir}/bak.zip", "r") as zf:
        print(f"file list: {zf.namelist()}")

        report_f = zf.getinfo("report.pdf")
        print(
            f"file size: {report_f.file_size}, compress size: {report_f.compress_size}"
        )


def test_unzip_files():
    # unzip file
    with zipfile.ZipFile(f"{tmp_dir}/bak.zip", "r") as zf:
        zf.extract("data.csv", path=f"{tmp_dir}/output")

    # unzip all files
    with zipfile.ZipFile(f"{tmp_dir}/bak.zip", "r") as zf:
        zf.setpassword(b"xxxx")
        zf.extractall(f"{tmp_dir}/extra")


if __name__ == "__main__":
    pass
