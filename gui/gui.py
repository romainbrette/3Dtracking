from tkinter import simpledialog
import tkinter as tk

__all__ = ['ParametersDialog']

class ParametersDialog(simpledialog.Dialog):
    '''
    A GUI to enter parameter values (only numerical).
    The parameters are a list of tuples (parameter name, text, default value)
    '''
    def __init__(self, parent=None, title=None, parameters=None):
        if parent is None:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            parent = root
        self.parameters = parameters
        self.parameters = parameters
        self.value = {}
        self.checks = {}
        self.type = {x: type(z) for x, _, z in parameters}
        super().__init__(parent, title)  # Call the parent class initializer

    def body(self, master):
        i = 0
        self.entry = {}
        for name, text, default in self.parameters:
            # Create labels and entry fields for fps and pixelsize
            tk.Label(master, text=text).grid(row=i, column=0)

            if self.type[name] == type(True):
                self.checks[name] = tk.BooleanVar(master, default)
                self.entry[name] = tk.Checkbutton(master, variable=self.checks[name])
                if default:
                    self.entry[name].select()
            else:
                self.entry[name] = tk.Entry(master)
                # Set default values
                self.entry[name].insert(0, str(default))

            self.entry[name].grid(row=i, column=1)
            i += 1

        return self.entry[self.parameters[0][0]]  # Initial focus

    def apply(self):
        # Retrieve the entered values when OK is clicked
        for name, text, default in self.parameters:
            if self.type[name] == type(True):
                self.value[name] = self.checks[name].get()
            else:
                self.value[name] = self.type[name](self.entry[name].get())
