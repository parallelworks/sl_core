from client import Client
import sys, json
from time import sleep
from datetime import datetime

def printd(*args):
    print(datetime.now(), *args)


def start_resource(resource_name, c):
    printd('Starting resource {}'.format(resource_name))
    # check if resource exists and is running already
    resource = c.get_resource(resource_name)
    if resource:
        if resource['status'] == "off":
            c.start_resource(resource_name)
            printd('{} started'.format(resource_name))
            return 'started'
        else:
            printd('{} already running'.format(resource_name))
            return 'already-running'
    else:
        printd('{} not found'.format(resource_name))
        return 'not-found'

def stop_resource(resource_name, c):
    printd('Stopping resource {}'.format(resource_name))
    # check if resource exists and is stopped already
    resource = c.get_resource(resource_name)
    if resource:
        if resource['status'] == "off":
            printd('{} already stopped'.format(resource_name))
            return 'already-stopped'
        else:
            c.stop_resource(resource_name)
            printd('{} stopped'.format(resource_name))
            return 'stopped'
    else:
        printd('{} not found'.format(resource_name))
        return 'not-found'


def launch_workflow(wf_name, wf_xml_args, user, c):
    printd('Launching workflow {wf} in user {user}'.format(
        wf = wf_name,
        user = user
    ))
    printd('XML ARGS: ', json.dumps(wf_xml_args, indent = 4))
    jid,djid = c.start_job(wf_name, wf_xml_args, user)
    return jid, djid


def wait_workflow(djid, wf_name, c):
    printd('Waiting for workflow', wf_name)
    while True:
        try:
            state = c.get_job_state(djid)
        except:
            state = 'starting'

        if state in ['ok', 'deleted', 'error']:
            return state

        printd('Workflow', wf_name, 'state:', state)
        sleep(10)

    printd(wf_name, 'completed successfully')