from project4.power_method import power_method
from project4.load_data import loaddata
from time import process_time

def problem1():
    P = loaddata()

    time_pm_085 = process_time()
    x085, i085 = power_method(P, 0.85, 0.00000001)
    time_pm_085 = process_time() - time_pm_085

    print('Power method a=0.85: Time {}s for {} iterations'.format(time_pm_085, i085))
    print(x085[:100])

if __name__=='__main__':
    problem1()