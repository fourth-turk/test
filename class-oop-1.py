# python oop classes

class Employee:

   num_of_emps = 0
   raise_amount = 1.04
   
   def __init__(self, first, last, pay):
      self.first = first
      self.last = last
      self.pay = pay
      self.email = first + '.' + last + '@email.com'

      Employee.num_of_emps += 1

   # setting property we change a method to attribute
   @property
   def email2(self):
      return '{}.{}@email.com'.format(self.first, self.last)

   @property
   def fullname(self):
      return '{} {}'.format(self.first, self.last)

   @fullname.setter
   def fullname(self, name):
      first, last = name.split(' ')
      self.first = first
      self.last = last

   @fullname.deleter
   def fullname(self):
      print('Delete Name')
      self.first = None
      self.last = None

   def apply_raise(self):
      self.pay = int(self.pay * self.raise_amount)

   @classmethod
   def set_raise_amt(cls,amount):
      cls.raise_amount = amount

   # alternative constructor
   @classmethod
   def from_string(cls, emp_str):
      first, last, pay = emp_str.split('-')
      return cls(first, last, pay)

   # when not using any self, cls variables, we just want some info
   @staticmethod
   def is_workday(day):
      if day.weekday() == 5 or day.weekday() == 6:
         return False
      return True

   # special methods - magic functions
   # for dev user
   def __repr__(self): 
      return "Employee('{}', '{}', '{}')".format(self.first, self.last, self.pay)

   # for end user
   def __str__(self):
      return '{} - {}'.format(self.fullname(), self.email)

   def __add__(self, other):
      return self.pay + other.pay

   def __len__(self):
      return len(self.fullname())


# subclass, inherits from class Employee
class Developer(Employee):
   raise_amount = 1.1

   def __init__(self, first, last, pay, prog_lang):
      # super().__init__(first, last, pay) # pass to init of Employee class
      Employee.__init__(self, first, last, pay) # similar 
      self.prog_lang = prog_lang


class Manager(Employee):
   
   def __init__(self, first, last, pay, employees=None):
      Employee.__init__(self, first, last, pay) # similar 
      if employees is None:
         self.employees = [] # bad to have mutable datatypes as default arguments therefore None above
      else:
         self.employees = employees

   def add_emp(self, emp):
      if emp not in self.employees:
         self.employees.append(emp)

   def remove_emp(self, emp):
      if emp in self.employees:
         self.employees.remove(emp)

   def print_emps(self):
      for emp in self.employees:
         print('-->', emp.fullname())


dev_1 = Developer('corey', 'mee', 50000, 'python')
dev_2 = Developer('test', 'bob', 60000, 'java')

# print(help(Developer))

# emp_str_1 = 'john-doe-70000'

# new_emp_1 = Employee.from_string(emp_str_1)
# print(new_emp_1.email)
# print(new_emp_1.pay)

# import datetime
# my_date = datetime.date(2016,7,11)
# print(Employee.is_workday(my_date))

# print(dev_1.email)

# print(dev_1.pay)
# dev_1.apply_raise()
# print(dev_1.pay)


# mgr_1 = Manager('sue', 'smith', 9000, [dev_1])

# print(mgr_1.email)

# mgr_1.print_emps()
# mgr_1.add_emp(dev_2)
# print()
# mgr_1.print_emps()
# print()
# mgr_1.remove_emp(dev_1)
# mgr_1.print_emps()



# print(isinstance(mgr_1, Developer))
# print(issubclass(Developer, Employee))
# print(issubclass(Manager, Developer))

###5 specila / magic / dunner functions

# print(dev_1)
# print(dev_1 + dev_2)

# print(len('test'))
# print('test'.__len__())
# print(len(dev_1))

###6 property decorators: getter, setter, deleter

print(dev_1.first, dev_1.last)

dev_1.first = 'jim'
print(dev_1.first, dev_1.last)

dev_1.fullname = 'bob hope'

print(dev_1.email)
print(dev_1.email2)
print(dev_1.fullname)

print(dev_1.first, dev_1.last)

emp_1 = Employee('john', 'smith', 40000)
# print(emp_1.first)
# print(emp_1.fullname)

emp_1.fullname = 'corey shafer'
print(emp_1.first) # first should be corey not john, sth wong.. ah, using v2.7
print(emp_1.fullname)
del emp_1.fullname

import sys
print(sys.version_info)



