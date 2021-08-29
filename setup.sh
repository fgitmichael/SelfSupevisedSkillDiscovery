mkdir ../rlkitdiayn
git clone https://github.com/johnlime/RlkitExtension ../rlkitdiayn
pip install -e ../rlkitdiayn

pushd ..
mkdir pybulletgym
cd pybulletgym
git clone https://github.com/benelot/pybullet-gym .
pip install -e .
git checkout 1625352a4000cf861eb1fced1cb234af12e5f7d4
popd

pip install
