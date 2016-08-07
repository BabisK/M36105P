from project4.power_method import power_method
from project4.load_data import loaddata
from time import process_time

def problem2():
    P = loaddata()

    time_pm_099 = process_time()
    x099, i099 = power_method(P, 0.99, 0.00000001)
    time_pm_099 = process_time() - time_pm_099

    print('Power method a=0.99: Time {}s for {} iterations'.format(time_pm_099, i099))
    print(x099[:100])

if __name__=='__main__':
    problem2()