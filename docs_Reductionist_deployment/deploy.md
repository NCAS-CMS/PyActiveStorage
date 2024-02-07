### How to deploy [Reductionist](https://github.com/stackhpc/reductionist-rs) on a Rocky 9 cloud machine

- 99% of the docs are found in Reductionist's nice [deployment instructions](https://stackhpc.github.io/reductionist-rs/deployment/)
- there are a few caveats specific to a pristine Rocky 9 (and other distros) deployment though:
  - (n00b step) always have a system `pip` by installing it with: `python -m ensurepip --upgrade`
  - system Python executable is `python3` - you can, of course, `ln -s` it to `python`, or, better, run Ansible pointing it to the correct system Python3: `ansible-playbook -i reductionist-rs/deployment/inventory reductionist-rs/deployment/site.yml -e 'ansible_python_interpreter=/usr/bin/python3'`
  - that call *may result* (as in our case) in a barf:
    ```
    TASK [Ensure step RPM is installed] **************************************************************************************************** 
fatal: [localhost]: FAILED! => {"changed": false, "msg": "Failed to validate GPG signature for step-cli-0.24.4-1.x86_64: Package step-cli_0.24.4_amd643z16ickc.rpm is not signed"}
    ```
  - that's because, in our case, we missed the `step-cli` package, and a `dfn` install is not well liked by the system (it's not `mamba` heh)  - that gets sorted out via [Step's install docs](https://smallstep.com/docs/step-cli/installation):
    ```
    wget https://dl.smallstep.com/cli/docs-cli-install/latest/step-cli_amd64.rpm
    sudo rpm -i step-cli_amd64.rpm
    ```
