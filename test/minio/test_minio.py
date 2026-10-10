from minio import Minio
from minio.error import S3Error

def main():
    # 1. 实例化MinIO客户端
    client = Minio("192.168.253.140:9000",
                   access_key="minioadmin",
                   secret_key="minioadmin",
                   secure=False
                   )

    # 2. 上传的文件地址
    source_file = r"C:\\Users\\a3384310973\\Desktop\\img1.jpg"

    # 3. 桶名`
    bucket_name = "python-test-bucket"

    # 4. 对象名字
    destination_file = "my-test-png.jpg"

    # 5. 判断桶是否存在
    found = client.bucket_exists(bucket_name)
    if not found:
        # 5.1 创建桶
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    # 6. 上传文件
    client.fput_object(
        bucket_name, destination_file, source_file,
    )
    print("上传成功")


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)
