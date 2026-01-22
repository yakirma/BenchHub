import tkinter as tk
from tkinter import filedialog, messagebox
import base64
import os

class ObfuscatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DLP-Safe Code Obfuscator")
        self.root.geometry("400x250")
        
        self.label = tk.Label(root, text="Select a file to obfuscate", pady=20)
        self.label.pack()
        
        self.select_btn = tk.Button(root, text="Select File", command=self.select_file, width=20)
        self.select_btn.pack(pady=10)
        
        self.status = tk.Label(root, text="", fg="gray")
        self.status.pack(pady=10)
        
        self.file_path = None

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Text/Code Files", "*.txt *.py"), ("All Files", "*.*")]
        )
        if self.file_path:
            self.obfuscate_and_save()

    def obfuscate_and_save(self):
        try:
            with open(self.file_path, "rb") as f:
                content = f.read()
            
            encoded = "BASE64:".encode("utf-8") + base64.b64encode(content)
            
            save_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                initialfile=os.path.basename(self.file_path).replace(".py", ".safe.txt"),
                filetypes=[("Text Files", "*.txt")]
            )
            
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(encoded)
                messagebox.showinfo("Success", f"File saved to:\n{save_path}")
                self.status.config(text="Done!", fg="green")
            else:
                self.status.config(text="Save cancelled", fg="orange")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to obfuscate:\n{str(e)}")
            self.status.config(text="Error occurred", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObfuscatorApp(root)
    root.mainloop()
