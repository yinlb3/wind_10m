#!user/bin.python3

"""
常用工具函数模块

Founded in 2026-04-03
Modified in 2026-04-03
@author: yinlb
"""


def format_time(second: float, is_abbreviation: bool = False) -> str:
    """
    将秒数格式化为易读的时间字符串
    
    用于程序运行时间统计等场景，支持缩写和完整两种格式。
    
    参数:
        second: 秒数，必须为非负数
        is_abbreviation: 是否使用缩写格式, 默认为False
            - True: 返回 '45.5s', '5m', '2h' 等缩写格式
            - False: 返回 '45.5 seconds', '5 minutes', '2 hours' 等完整格式
    
    返回:
        格式化后的时间字符串
    
    异常:
        ValueError: 当输入秒数为负数时抛出
    
    示例:
        >>> format_time(45.5)
        '45.5 seconds'
        >>> format_time(45.5, is_abbreviation=True)
        '45.5s'
        >>> format_time(3600)
        '60.0 minutes'
        >>> format_time(7200, is_abbreviation=True)
        '2.0h'
    """
    if second < 0:
        raise ValueError('Input seconds cannot be negative')
    
    if is_abbreviation:
        if second <= 60:
            time_str = str(second) + 's'
        elif second <= 3600:
            time_str = str(second / 60) + 'm'
        else:
            time_str = str(second / 3600) + 'h'
    else:
        if second <= 1:
            time_str = str(second) + ' second'
        elif second <= 60:
            time_str = str(second) + ' seconds'
        elif second <= 3600:
            time_str = str(second / 60) + ' minutes'
        else:
            time_str = str(second / 3600) + ' hours'
    
    return time_str
