//
// This file is part of an OMNeT++/OMNEST simulation example.
//
// Copyright (C) 1992-2015 Andras Varga
//
// This file is distributed WITHOUT ANY WARRANTY. See the file
// `license' for details on this and other legal matters.
//

#include <omnetpp.h>

using namespace omnetpp;

/**
 * Sends received packets to one of the outputs; see NED file for more info
 */
class Switch : public cSimpleModule
{
  private:
    short int priority;
    int numGates;
    double retain;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
};

Define_Module(Switch);

void Switch::initialize()
{
    priority = par("priority");
    numGates = gateSize("out");
    retain = par("retain");
}

void Switch::handleMessage(cMessage *msg)
{
    int chance = uniform(0.0, 1.0);
    int gate = priority;

    if (chance > retain) {
        gate = intuniform(0, numGates-1);
    }
    msg->setSchedulingPriority(priority);
    send(msg, "out", gate);
}

