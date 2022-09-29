#pragma once

namespace CollisionExample {

Engine::Engine(CollisionSim *sim, madrona::WorkerInit &&init)
    : madrona::CustomContext<Engine>(std::move(init)),
      sim_(sim)
{}

}
