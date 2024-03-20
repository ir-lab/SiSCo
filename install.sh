#! /bin/bash


# exit on any error
set -e

pip install --upgrade pip 
pip install -r requirements.txt
echo "Finished installing python packages"

# Function to install a Git repository with CMake
install_repo() {
  repo_url=$1
  branch=$2
  repo_name=$(basename $repo_url .git)
  repo_realpath=$(realpath "$repo_name")

  if [ -d "$repo_name" ]; then
    echo "$repo_name already cloned."
  else
    echo "Cloning $repo_name ..."
    git clone $repo_url
  fi

  echo "Building and installing $repo_name ..."
  cd $repo_name && git checkout $branch

  if [ $repo_name == "orocos_kinematics_dynamics" ]; then
    read -p $'\nEnter \"y\" if you are inside python virtual environment else press \"enter\" to stop installation\n' user_input
    if [[ $user_input != "y" ]]; then
        echo -e "Please source the python virtual environment"
        exit 0
    fi

    git submodule update --init
    echo "install kdl lib"
    mkdir orocos_kdl/build && cd orocos_kdl/build && cmake ..
    make -j$(nproc)
    sudo make install
    cd  ../..
    echo "installing pykdl lib"
    mkdir python_orocos_kdl/build && cd python_orocos_kdl/build && cmake ..
    make -j$(nproc)
    sudo make install
    cd  ../..
    read -p $'Enter virtual environment path e.g. /home/user/.sisco\n' vm_path
    if [ -d "$vm_path" ]; then
        echo "Copying compiled lib PyKDL.so to $vm_path/lib/python3.8/site-packages"
        cp /usr/local/lib/python3/dist-packages/PyKDL.so $vm_path/lib/python3.8/site-packages

    else
        echo "Invalid virtual env path $vm_path" 
        exit 0

    fi
  else
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    sudo make install
    cd ../..
  fi

  echo "Removing $repo_realpath ..."
  rm -rf $repo_realpath

  echo "$repo_name installed and removed."
}

echo "Installing PyKDL"
install_repo https://github.com/orocos/orocos_kinematics_dynamics.git pykdl_tree
 
