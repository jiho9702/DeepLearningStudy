print(type(10))        #int

print(type(2.718))     #float

print(type('hello'))    #str

x = 10
print(x)

x = 100
print(x)

y = 3.14
print(x*y)

print(type(x*y))

a = [1,2,3,4,5]
print(a)                #리스트 내용 출력
print(len(a))           #리스트 길이
print(a[0])             #첫번째 원소 접근
print(a[4])             #다섯번째 원소 접근
a[4] = 99               #다섯번째 원소에 접근 해 값 대입
print(a)

print(a[0:2])           #인덱스 0부터 2까지 얻기(2는 포합 X)
print(a[1:])            #인덱스 1부터 끝까지 얻기
print(a[:3])            #인덱스 처음부터 3까지 얻기(3은 미포함)
print(a[:-1])           #인덱스 마지막원소의 1개 앞까지 얻기
print(a[:-2])           #인덱스 마지막원소의 2개 앞까지 얻기

me = {'height' : 180}   #딕셔너리 생성
print(me['height'])     #원소에 접근  출력 : 180
me['weight'] = 70       #새로운 원소 추가
print(me)               #출력 : weight = 70, height : 180


hungry = True
sleepy = False

print(type(hungry))
print(not hungry)       #출력 : False
print(hungry and sleepy)
print(hungry or sleepy) #논리연산을 쉽게 할수 있다.

hungry = True
if hungry:
    print("I'm humngry")

hungry = False
if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry")
    print("I'm sleepy")

for i in [1,2,3]:
    print(i)

def hello():
    print("Hello World")

hello()

def hello(object):
    print("Hello " + object + "!")

hello('cat')