from os import system

def main():
  system("cd frontend && yarn deploy && cd ../ && rm -r public/ && mv ./frontend/build ./public")

main()