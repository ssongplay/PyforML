import numpy as np

######################################################################
# problem 1
# n의 제곱수로 2 dimentional array를 생성하는 ndarray.

def n_size_ndarray_creation(n, dtype=np.int8):
    return np.array(range(n**2), dtype).reshape(n,n)

# 실행결과
print(n_size_ndarray_creation(3))
print()


######################################################################
# problem 2
# shape이 지정된 크기의 ndarray를 생성,
# 이때 행렬의 element는 type에 따라 0, 1 또는 empty로 생성됨
# type : 생성되는 element들의 값을 지정함
# 0은 0, 1은 1, 99는 empty 타입으로 생성됨

def zero_or_one_or_empty_ndarray(shape, type=0, dtype=np.int8):
    if type == 0:
        return np.zeros(shape, dtype)
    elif type == 1:
        return np.ones(shape, dtype)
    elif type == 99:
        return np.empty(shape, dtype)

# 실행 결과
print(zero_or_one_or_empty_ndarray(shape=(2,2), type=1))
print()
print(zero_or_one_or_empty_ndarray(shape=(3,3), type=99))
print()


######################################################################
# problem 3
# 입력된 ndarray X를 n_row의 값을 row의 개수로 지정한 matrix를 반환함.
# 이때 입력하는 X의 size는 2의 거듭제곱수로 전제함.
# 만약 n_row과 1일 때는 matrix가 아닌 vector로 반환함.

def change_shape_of_ndarray(X, n_row):
    # 내가 짠 코드 : return X.reshape(n_row, -1) 이지만
    # n_row가 1일 때는 matrix가 아닌 vector로 반환해야하므로 flatten() 써줘야 함
    return X.flatten() if n_row == 1 else X.reshape(n_row, -1)

# 실행 결과
X = np.ones((32,32), dtype=np.int8)
print(change_shape_of_ndarray(X, 1))
print()
print(change_shape_of_ndarray(X, 512))
print()


######################################################################
# problem 4
# 입력된 ndarray X_1과 X_2를 axis로 입력된 축을 기준으로 통합하여 반환하는 함수
# X_1과 X_2는 matrix 또는 vector 임, 그러므로 vector 일 경우도 처리할 수 가 있어야 함
# axis를 기준으로 통합할 때, 통합이 불가능하면 False가 반환됨.
# 단 X_1과 X_2 Matrix, Vector 형태로 들어왔다면, Vector를 row가 1개인 Matrix로 변환하여 통합이 가능한지 확인할 것
def concat_ndarray(X_1, X_2, axis):
    # 강의영상 참고
    try : # vector형태로 들어왔다면 vector를 row가 1개인 matrix로 변경
        if X_1.ndim == 1:
            X_1 = X_1.reshape(1, -1)
        if X_2.ndim == 1:
            X_2 = X_2.reshape(1, -1)
        return np.concatenate((X_1, X_2), axis)
    # concat을 해서 안되면 value 에러가 남
    except ValueError as e:
        return False

# 실행 결과
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
print(concat_ndarray(a, b, 0))
print()
print(concat_ndarray(a, b, 1))
print()

a = np.array([1, 2])
b = np.array([5, 6, 7])
print(concat_ndarray(a, b, 1))
print()
print(concat_ndarray(a, b, 0))


######################################################################
# problem 5
# 입력된 Matrix 또는 Vector를 ndarray X의 정규화된 값으로 변환하여 반환함
# 이때 정규화 변환 공식 Z = (X - X의평균) / X의 표준편차로 구성됨.
# X의 평균과 표준편차는 axis를 기준으로 axis 별로 산출됨.
# Matrix의 경우 axis가 0 또는 1일 경우, row 또는 column별로 Z value를 산출함.
# xis가 99일 경우 전체 값에 대한 normalize 값을 구함.

def normalize_ndarray(X, axis=99, dtype=np.float32):
    # 어려워요 ㅠㅠ 강의 참고
    X = X.astype(np.float32)  # X를 float 형으로 바꾸기
    n_row, n_column = X.shape  # tuple형태의 X.shape 를 unpacking
    if axis == 99 :  # 전체 값에 대한 normalize 값 구하기
        x_mean = np.mean(X)
        x_std = np.std(X)
        Z = (X - x_mean) / x_std
    if axis == 0 :  # row 별로 Z value를 산출
        x_mean = np.mean(X, 0).reshape(1, -1)
        x_std = np.std(X, 0).reshape(1, -1)
        Z = (X - x_mean) / x_std
    if axis == 1:
        x_mean = np.mean(X, 1).reshape(n_row, -1)
        x_std = np.std(X, 1).reshape(n_row, -1)
        Z = (X - x_mean) / x_std
    return Z

# 실행 결과
X = np.arange(12, dtype=np.float32).reshape(6,2)
print(normalize_ndarray(X))
print()
print(normalize_ndarray(X, 1))
print()
print(normalize_ndarray(X, 0))
print()


######################################################################
# problem 6
# 입력된 ndarray X를 argument filename으로 저장함

def save_ndarray(X, filename="test.npy"):
    # 그대로 따라해도 오류가 남..
    pass


######################################################################
# problem 7
# 입력된 ndarray X를 String type의 condition 정보를 바탕으로 해당 컨디션에 해당하는 ndarray X의 index 번호를 반환함
# 단 이때, str type의 조건인 condition을 코드로 변환하기 위해서는 eval(str("X") + condition)를 사용할 수 있음
def boolean_index(X, condition):
    condition = eval(str("X") + condition)
    return np.where(condition)

# 실행 결과
X = np.arange(32, dtype=np.float32).reshape(4, -1)
print(boolean_index(X, "== 3"))
print()
X = np.arange(32, dtype=np.float32)
print(boolean_index(X, "> 6"))
print()


######################################################################
# problem 8
# 입력된 vector type의 ndarray X에서 target_value와 가장 차이가 작게나는 element를 찾아 리턴함
# 이때 X를 list로 변경하여 처리하는 것은 실패로 간주함.
# X: 입력하는 vector type의 ndarray
# target_value : 가장 유사한 값의 기준값이 되는 값

def find_nearest_value(X, target_value):
    return X[np.argmin(np.abs(X - target_value))]

# 실행 결과
X = np.random.uniform(0, 1, 100)
target_value = 0.3
print(find_nearest_value(X, target_value))
print()


######################################################################
# problem 9
# 입력된 vector type의 ndarray X에서 큰 숫자 순서대로 n개의 값을 반환함.

def get_n_largest_values(X, n):
    # np.argsort 처음봄 -> 정리하기
    return X[np.argsort(X[::1]) [:n]]

# 실행 결과
X = np.random.uniform(0, 1, 100)
print(get_n_largest_values(X, 3))
print()
print(get_n_largest_values(X, 5))
print()
