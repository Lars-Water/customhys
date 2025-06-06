import typing

def requires_main_process(func):
    """
    Decorator to mark a method as stateful and requiring execution
    on the single, authoritative object in the main process.
    Any method NOT marked with this will be executed locally in the child process
    if called through an RPCProxy.
    """
    func._requires_main_process = True
    return func

def replicated_to_parent(cls):
    """
    Class decorator for objects that live in a child process but whose state
    needs to be replicated back to the parent.
    It wraps the __setattr__ method to automatically push changes for
    attributes marked with the @replicate decorator.
    """
    original_setattr = cls.__setattr__

    def wrapped_setattr(self, name, value):
        # Call the original __setattr__ to actually set the value
        original_setattr(self, name, value)
        
        # Check if this attribute is marked for replication and push the update
        if name in getattr(self, '_replicated_attributes', []):
            if hasattr(self, 'rpc_context') and self.rpc_context:
                # IMPORTANT: Only send attribute updates AFTER the initial shadow copy is sent.
                # This prevents a race condition where updates arrive before the shadow object exists.
                if self.rpc_context._initial_shadow_copy_sent:
                    self.rpc_context.update_parent_attribute(name, value)
                
    cls.__setattr__ = wrapped_setattr
    return cls

def replicate(name_of_attribute):
    """
    A decorator to be used on a class to mark an attribute for replication.
    """
    def decorator(cls):
        if not hasattr(cls, '_replicated_attributes'):
            cls._replicated_attributes = []
        cls._replicated_attributes.append(name_of_attribute)
        return cls
    return decorator

def callable_from_main(func):
    """
    Decorator to mark a method on a child's object as safe to be
    called by the main process on the shadow copy.
    This implies the method is read-only and does not modify state.
    """
    func._is_callable_from_main = True
    return func

class RPCProxy:
    def __init__(self, local_object_copy: typing.Any, rpc_context: typing.Any, path_prefix: tuple = ()):
        # Use __dict__ to avoid triggering __setattr__
        self.__dict__['_local_object'] = local_object_copy
        self.__dict__['_rpc_context'] = rpc_context
        self.__dict__['_path'] = path_prefix
        self.__dict__['_call_counter'] = 0

    def __getattr__(self, name: str) -> typing.Any:
        # Get the attribute from the local copied object first.
        # This lets us inspect it for decorators before deciding what to do.
        try:
            attr = getattr(self._local_object, name)
        except AttributeError:
            raise AttributeError(f"'{type(self._local_object).__name__}' object has no attribute '{name}' in its local copy.")

        # If the attribute is NOT callable (i.e., it's a property/variable) AND it's decorated,
        # we must fetch its value from the main process.
        if not callable(attr) and hasattr(attr, '_requires_main_process'):
            full_path_str = ".".join(self._path + (name,))
            # Make an RPC call to get the attribute's value.
            return self._rpc_context._call_service_rpc(full_path_str, None) # No refresh payload on attribute access

        # Otherwise (if it's a regular attribute or a method), return a new proxy
        # that wraps the next object in the chain. The decision to execute vs. call
        # will be made in __call__.
        return RPCProxy(attr, self._rpc_context, self._path + (name,))

    def __call__(self, *args, **kwargs) -> typing.Any:
        # The target method is the local object itself
        target_method = self._local_object
        
        # Check if the method is marked to run on the main process
        if hasattr(target_method, '_requires_main_process'):
            # Increment call counter for periodic refresh
            self.__dict__['_call_counter'] += 1
            
            full_path_str = ".".join(self._path)
            
            # Check for periodic refresh
            refresh_payload = None
            if self._call_counter % 10 == 0: # Refresh every 10 stateful calls
                refresh_payload = self._rpc_context.get_refresh_payload()

            return self._rpc_context._call_service_rpc(full_path_str, refresh_payload, *args, **kwargs)
        else:
            # Execute locally on the copied object
            return target_method(*args, **kwargs) 