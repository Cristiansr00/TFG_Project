from menus.menu_principal import menu_principal
from utils.paths import initialize_project_dirs

def main():
    initialize_project_dirs()
    menu_principal()

if __name__ == "__main__":
    main()
