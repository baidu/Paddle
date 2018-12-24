import os
import hashlib
import importlib
import paddle.fluid


files = [
    "paddle.fluid.average",
    "paddle.fluid.backward",
    "paddle.fluid.clip",
    "paddle.fluid.data_feeder",
    "paddle.fluid.executor",
    "paddle.fluid.initializer",
    "paddle.fluid.io",
    "paddle.fluid.layers",
    "paddle.fluid.metrics",
    "paddle.fluid.nets",
    "paddle.fluid.optimizer",
    "paddle.fluid.profiler",
    "paddle.fluid.recordio_writer",
    "paddle.fluid.regularizer",
    "paddle.fluid.transpiler"
]


def md5(doc):
    hash = hashlib.md5()
    hash.update(str(doc))
    return hash.hexdigest()


def get_module():
    for fi in files:
        fi_lib = importlib.import_module(fi)
        doc_function =getattr(fi_lib, "__all__")
        for api in doc_function:
            api_name = fi + "." + api
            try:
                doc_module = getattr(eval(api_name), "__doc__")
            except:
                pass
            doc_md5_code = md5(doc_module)
            if not os.path.getsize(doc_md5_file):
                doc_dict[api_name] = doc_md5_code
                print(doc_dict)
            else:
                try:
                    if doc_dict[api_name] != doc_md5_code:
                       return "FALSE"
                except:
                    return "FALSE" 


def doc_md5_dict(doc_md5_path):
    with open(doc_md5_path, "rb")as f:
        doc_md5 = f.read()
        doc_md5_dict = ast.literal_eval(doc_md5)
    return doc_md5_dict


if __name__ == "__main__":
    doc_dict = {}
    doc_md5_file = "doc_md5.txt"
    if not os.path.exists(doc_md5_file):
        os.mknod(doc_md5_file)    
    else:
        doc_dict = doc_md5_dict(doct_md5_file)
    get_module()
    if not os.path.getsize(doc_md5_file):
        with open(doc_md5_file, 'wb')as f:
            f.write(str(doc_dict))
