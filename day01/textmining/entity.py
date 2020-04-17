from  dataclasses import dataclass

@dataclass
class Entity:
    context: str
    fname: str
    target: str

    @property
    def context(self) -> str: return self._context
    @context.setter
    def context(self,context): self._context = context

    @property
    def fname(self) -> str: return self._fname

    @fname.setter
    def fname(self, fname): self._fname = fname

    @property
    def target(self) -> str: return self._target

    @target.setter
    def target(self, target): self._target = target
