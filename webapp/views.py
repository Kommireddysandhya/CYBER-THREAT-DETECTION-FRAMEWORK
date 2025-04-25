from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

from django.shortcuts import render
from .models import predict


def read_file(file_name):
    opened_file = open(file_name, 'r')
    lines_list = []
    for line in opened_file:
        line = line.split()
        lines_list.append(line)
    #print(lines_list)
    return lines_list



# Create your views here.
# Create your views here.  
def home(request):
	return render(request,'index.html')


def input(request):
    file_name = 'account.txt'
    name = request.POST.get('name')
    password = request.POST.get('password')
    account_list = read_file(file_name)
    print(name)
    print(password)
    for i in account_list:

        if i[0] == name  and i[1] == password:
            print(i[0])
            print(i[1])
            return render(request,'input.html')
        else:
            return HttpResponse('Wrong Password or Name', content_type='text/plain')


class_names = ['Blackhole','Diversion','Normal','Overflow','Portscan','TCP-SYN']

def output(request):
	algo=request.POST.get('algo')
	row=int(request.POST.get('row'))
	out=predict(algo,row)
	print(out)
	classes = class_names[int(out)]

	print(classes)
	return render(request,'output.html',{'out':classes})
