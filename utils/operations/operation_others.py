from .operation_string import str2bool


def auto_convert_args_to_bool(args):
    for key, value in vars(args).items():
        try:
            converted_value = str2bool(value, extended=False)
            setattr(args, key, converted_value)
        except:
            pass
    return args
